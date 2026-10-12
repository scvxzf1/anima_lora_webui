"""CPU contracts for disposable full block VJP evidence."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime import anima_block_vjp_capture as capture
from bench.adaptive_runtime.probe_anima_train import validate_args
from networks.attention_dispatch import AttentionParams


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.fp32_residual = False
        self.gradient_checkpointing = True

    def _forward(self, x, emb, cross, attn, rope, adaln, use_fp32):
        assert attn.attn_mode == "torch"
        return x * self.weight + emb.reshape(1, 1, 1, 1, 1) + cross.mean() * 0

    def forward(self, x, emb_B_T_D, crossattn_emb, attn_params,
                rope_cos_sin=None, adaln_lora_B_T_3D=None, use_fp32=False):
        return checkpoint(self._forward, x, emb_B_T_D, crossattn_emb, attn_params,
                          rope_cos_sin, adaln_lora_B_T_3D, use_fp32, use_reentrant=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(14)] + [Block()])


class MultiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(28)])


def signature(scale=1, precision="fp32-reference"):
    return {"steps": 1, "sigmas": [0.5], "block_vjp_scope": capture.SCOPE,
            "block_vjp_helper_sha256": "a" * 64, "loss_scale": scale,
            "precision": precision, "weights_sha256": "b" * 64,
            "probe_sha256": "c" * 64}


def network():
    adapter = nn.Linear(1, 1, bias=False)
    adapter.lora_name = "lora_unet_blocks_14_mlp_layer1"
    adapter.lora_up = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(adapter.lora_up.weight)
    return SimpleNamespace(unet_loras=[adapter])


def multi_network():
    adapters = []
    for index in capture.MULTI_INDICES:
        adapter = nn.Linear(1, 1, bias=False)
        adapter.lora_name = f"lora_unet_blocks_{index}_mlp_layer1"
        adapter.lora_up = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(adapter.lora_up.weight)
        adapters.append(adapter)
    return SimpleNamespace(unet_loras=adapters)


def multi_signature(scale=1):
    return {**signature(scale), "block_vjp_scope": capture.MULTI_SCOPE,
            "block_vjp_indices": list(capture.MULTI_INDICES)}


def make_multi_capture(tmp_path, monkeypatch, *, scale=1):
    monkeypatch.setattr(capture, "SHAPE", (1, 1, 2, 3, 4))
    model, net = MultiModel(), multi_network()
    directory = tmp_path / "block-vjp"
    evidence = capture.MultiBlockVJPCapture(directory, model, net,
                                            signature=multi_signature(scale), uuid="GPU-test",
                                            scale=scale, replay={"state_sha256": "d" * 64})
    x = torch.arange(24, dtype=torch.float32).reshape(capture.SHAPE).requires_grad_()
    params = AttentionParams(attn_mode="torch", attention_mask=torch.ones(1, 4, dtype=torch.bool))
    y = x
    try:
        for index in capture.MULTI_INDICES:
            y = model.blocks[index](y, torch.ones(1, 1, 1), torch.ones(1, 4, 4), params,
                                    rope_cos_sin=(torch.ones(2, 2), torch.zeros(2, 2)),
                                    adaln_lora_B_T_3D=torch.ones(1, 1, 3, 4), use_fp32=True)
        (y.square().sum() * scale).backward()
        evidence.close()
    finally:
        evidence.restore()
    return model, x, y, directory


def make_capture(tmp_path, monkeypatch, *, scale=1, attn=None):
    monkeypatch.setattr(capture, "SHAPE", (1, 1, 2, 3, 4))
    model = Model()
    net = network()
    evidence = capture.BlockVJPCapture(tmp_path / "block-vjp", model, net,
                                       signature=signature(scale), uuid="GPU-test",
                                       scale=scale, replay={"state_sha256": "d" * 64})
    x = torch.arange(24, dtype=torch.float32).reshape(capture.SHAPE).requires_grad_()
    emb = torch.ones(1, 1, 1)
    cross = torch.ones(1, 4, 4)
    params = attn or AttentionParams(attn_mode="torch", softmax_scale=0.125,
                                      attention_mask=torch.ones(1, 4, dtype=torch.bool))
    try:
        y = model.blocks[14](x, emb, cross, params,
                             rope_cos_sin=(torch.ones(2, 2), torch.zeros(2, 2)),
                             adaln_lora_B_T_3D=torch.ones(1, 1, 3, 4), use_fp32=True)
        (y.square().sum() * scale).backward()
        evidence.close()
    finally:
        evidence.restore()
    return x, y, tmp_path / "block-vjp"


def test_checkpoint_full_hook_is_nonperturbing_and_unscales(tmp_path, monkeypatch):
    x, y, directory = make_capture(tmp_path, monkeypatch, scale=1024)
    manifest, tensors = capture.read_capture(directory)
    assert manifest["block_index"] == 14
    assert torch.allclose(tensors["g"], 2 * y.detach())
    assert torch.allclose(tensors["gx"], 4 * y.detach())
    assert torch.allclose(x.grad, 4096 * y.detach())
    kwargs = capture.conditioning(manifest, tensors, "cpu")
    assert kwargs["use_fp32"] is True
    assert kwargs["attn_params"].attention_mask.dtype == torch.bool
    assert torch.equal(kwargs["rope_cos_sin"][0], torch.ones(2, 2))
    assert capture.state_digests(Model(), network(), 14)[0] == manifest["base_state_sha256"]


def test_unknown_attention_object_and_duplicate_forward_fail(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown attention"):
        capture._pack_conditioning((torch.ones(1), torch.ones(1, 1, 1),
                                    torch.ones(1, 4, 4), object()), {})
    x, y, _directory = make_capture(tmp_path, monkeypatch)
    assert y.shape == x.shape


def test_hash_shape_missing_and_budget_fail_closed(tmp_path, monkeypatch):
    _, _, directory = make_capture(tmp_path, monkeypatch)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensor_shapes"].pop("gx")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape metadata"):
        capture.read_capture(directory)
    manifest["tensor_shapes"]["gx"] = list(capture.SHAPE)
    manifest_path.write_text(json.dumps(manifest))
    with (directory / "capture.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash/budget"):
        capture.read_capture(directory)
    monkeypatch.setattr(capture, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="budget"):
        make_capture(tmp_path / "small", monkeypatch)


def test_probe_config_rejected_before_gpu(tmp_path):
    args = SimpleNamespace(output=tmp_path / "result.json", steps=1, swap=0, loss_scale=1,
                           precision="fp32-reference", fp32_pattern=[], fp16_only_pattern=[],
                           record_replay=True, replay_reference=None, single_sigma=0.4,
                           capture_block_vjp=True, capture_local_vjp=False,
                           capture_block_boundaries=False, backward_counterfactual_suite=False,
                           local_vjp_profile="linear", base_backward_mode="native")
    with pytest.raises(ValueError, match="Block VJP"):
        validate_args(args)


def test_actual_block_policy_is_checked_before_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "SHAPE", (1, 1, 2, 3, 4))
    model = Model()
    model.blocks[14].fp32_residual = True
    with pytest.raises(ValueError, match="Actual block residual"):
        capture.BlockVJPCapture(tmp_path / "wrong", model, network(),
                               signature=signature(), uuid="GPU-test", scale=1,
                               replay={"state_sha256": "d" * 64})
    assert not (tmp_path / "wrong").exists()


def test_multi_block_full_checkpoint_capture_and_unscale(tmp_path, monkeypatch):
    model, x, y, directory = make_multi_capture(tmp_path, monkeypatch, scale=1024)
    root, blocks = capture.read_multi_capture(directory)
    assert root["block_indices"] == [15, 17, 27]
    assert set(blocks) == {15, 17, 27}
    assert root["tensor_bytes"] == sum(item[0]["tensor_bytes"] for item in blocks.values())
    assert torch.allclose(x.grad, 16 * y.detach() * 1024)
    for index, (manifest, tensors) in blocks.items():
        assert manifest["block_index"] == index
        multiplier = {15: 8, 17: 4, 27: 2}[index]
        assert torch.allclose(tensors["g"], multiplier * y.detach())
        assert torch.allclose(tensors["gx"], 2 * multiplier * y.detach())
        assert tensors["g"].data_ptr() != tensors["gx"].data_ptr()
        kwargs = capture.conditioning(manifest, tensors, "cpu")
        assert kwargs["attn_params"].attention_mask.dtype == torch.bool
        assert kwargs["crossattn_emb"].shape == (1, 4, 4)
        assert len(model.blocks[index]._forward_hooks) == 0
        assert len(model.blocks[index]._backward_hooks) == 0


def test_multi_capture_missing_child_and_root_tamper_fail_closed(tmp_path, monkeypatch):
    _, _, _, directory = make_multi_capture(tmp_path, monkeypatch)
    root_file = directory / "manifest.json"
    root = json.loads(root_file.read_text())
    root["blocks"]["17"]["sha256"] = "0" * 64
    root_file.write_text(json.dumps(root))
    with pytest.raises(ValueError, match="identity mismatch"):
        capture.read_multi_capture(directory)
    root["blocks"]["17"]["sha256"] = json.loads(
        (directory / "block-17" / "manifest.json").read_text())["sha256"]
    root_file.write_text(json.dumps(root))
    (directory / "block-17" / "capture.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        capture.read_multi_capture(directory)


def test_multi_indices_and_aggregate_budget_fail_before_complete(tmp_path, monkeypatch):
    for indices in ((15, 17), (15, 17, 17), (17, 15, 27), (14, 15, 17, 27)):
        with pytest.raises(ValueError, match="indices"):
            capture.validate_block_indices(indices)
    monkeypatch.setattr(capture, "SHAPE", (1, 1, 2, 3, 4))
    monkeypatch.setattr(capture, "MAX_BYTES", 1000)
    model = MultiModel()
    evidence = capture.MultiBlockVJPCapture(tmp_path / "small", model, multi_network(),
                                            signature=multi_signature(), uuid="GPU-test",
                                            scale=1, replay={"state_sha256": "d" * 64})
    x = torch.ones(capture.SHAPE, requires_grad=True)
    params = AttentionParams(attn_mode="torch")
    try:
        with pytest.raises(ValueError, match="aggregate budget"):
            for index in capture.MULTI_INDICES:
                x = model.blocks[index](x, torch.ones(1, 1, 1), torch.ones(1, 4, 4), params)
    finally:
        evidence.restore()
    assert json.loads((tmp_path / "small" / "manifest.json").read_text())["status"] == "capturing"


def test_probe_rejects_bad_multi_indices_before_gpu(tmp_path):
    args = SimpleNamespace(output=tmp_path / "result.json", steps=1, swap=0, loss_scale=1,
                           precision="fp32-reference", fp32_pattern=[], fp16_only_pattern=[],
                           record_replay=True, replay_reference=None, single_sigma=0.5,
                           capture_block_vjp=True, capture_local_vjp=False,
                           capture_block_boundaries=False, backward_counterfactual_suite=False,
                           local_vjp_profile="linear", base_backward_mode="native",
                           block_vjp_indices="15,27,17", weights=tmp_path / "weights",
                           inputs=tmp_path / "inputs")
    with pytest.raises(ValueError, match="indices"):
        validate_args(args)
    args.block_vjp_indices = "15,17,27"
    args.capture_block_vjp = False
    with pytest.raises(ValueError, match="requires"):
        validate_args(args)
