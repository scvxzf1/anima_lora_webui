"""Native-shape flatten invariants (replaces the retired static-pad path).

`compile_blocks()` is the single switch that turns on native-shape flattening:
the forward reshapes each bucket's patch grid to a fake-5D `(B, 1, seq_len, 1, D)`
so dynamo keys the block graph on token count alone. The reshape must be
*bit-exact* to the eager 5D path (the gap=0 control the old pad-leak probe
verified), and eager (uncompiled) forwards must skip it entirely.

The two token counts exercised correspond to the shipped CONSTANT_TOKEN_BUCKETS
families: 4032 (= 63·64) and 4200 (= 60·70).
"""

from __future__ import annotations

import types

import torch

from library.anima.models import Anima
from library.runtime.peak_probe import PeakProbe


def _tiny_anima(num_blocks: int = 2) -> Anima:
    """A small but real Anima DiT runnable on CPU."""
    model = Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    return model.eval()


def _inputs(latent_h: int, latent_w: int):
    """Inputs whose patchified token count is (h/2)*(w/2)."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


def test_compile_blocks_sets_native_flatten_and_budget():
    import torch._dynamo as _dynamo

    model = _tiny_anima()
    assert model._native_flatten is False  # off until compile_blocks

    _dynamo.config.cache_size_limit = 1  # force the max() to raise it
    model.compile_blocks(backend="eager")

    assert model._native_flatten is True
    # 2 token-count families → 2*2 + 8 = 12, and never lowered below current.
    assert _dynamo.config.cache_size_limit >= 12


def test_compile_blocks_does_not_lower_a_higher_budget():
    """The max() lets a multi-resolution caller (e.g. SPD) pre-raise the limit."""
    import torch._dynamo as _dynamo

    model = _tiny_anima()
    _dynamo.config.cache_size_limit = 64  # a caller asked for more headroom
    model.compile_blocks(backend="eager")
    assert _dynamo.config.cache_size_limit == 64


def test_compile_blocks_keeps_swapped_tail_eager(monkeypatch, capsys):
    compiled: list[object] = []

    def fake_compile(fn, **_kwargs):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _tiny_anima(num_blocks=4)
    model.blocks_to_swap = 2

    model.compile_blocks(backend="eager")

    assert len(compiled) == 2
    assert "_forward" in model.blocks[0].__dict__
    assert "_forward" in model.blocks[1].__dict__
    assert "_forward" not in model.blocks[2].__dict__
    assert "_forward" not in model.blocks[3].__dict__
    assert "2 resident compiled / 2 swapped (eager)" in capsys.readouterr().out


def test_compile_blocks_can_compile_swapped_tail(monkeypatch, capsys):
    compiled: list[object] = []

    def fake_compile(fn, **_kwargs):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _tiny_anima(num_blocks=4)
    model.blocks_to_swap = 2

    model.compile_blocks(backend="eager", compile_block_scope="all")

    assert len(compiled) == 4
    assert "_forward" in model.blocks[0].__dict__
    assert "_forward" in model.blocks[1].__dict__
    assert "_forward" in model.blocks[2].__dict__
    assert "_forward" in model.blocks[3].__dict__
    assert "2 resident + 2 swapped compiled" in capsys.readouterr().out


def test_compile_blocks_reuses_base_forward_on_recompile(monkeypatch):
    """A second compile_blocks (dynamic-seq range widened for a new sample
    resolution) must compile the ORIGINAL forward, not the already-compiled
    one — otherwise the graphs nest and the seq axis is re-marked against
    stale bounds."""
    compiled_sources = []

    def fake_compile(fn, **_kwargs):
        compiled_sources.append(fn)

        def compiled(*args, **kwargs):
            return fn(*args, **kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _tiny_anima()

    model.compile_blocks(backend="eager", n_token_families=2)
    base_forward = model.blocks[0]._anima_compile_base_forward
    first_compiled_forward = model.blocks[0]._forward

    model.compile_blocks(backend="eager", n_token_families=3)

    assert model.blocks[0]._anima_compile_base_forward is base_forward
    assert compiled_sources[0] is base_forward
    # second pass over block 0 (2 blocks per pass) — still the base, not the
    # first pass's compiled callable
    assert compiled_sources[2] is base_forward
    assert model.blocks[0]._forward is not first_compiled_forward


@torch.no_grad()
def test_compile_blocks_dynamic_seq_marks_range_and_runs():
    """dynamic_seq wraps the compiled inner, so eager backend still executes."""

    model = _tiny_anima()
    model.compile_blocks(
        backend="eager",
        dynamic_seq=True,
        bucket_resolutions=[(1008, 1024), (960, 1120)],
    )

    assert model._native_flatten is True
    assert model._dynamic_seq is True
    assert model._dynamic_seq_range == (4032, 4200)

    out = model.forward_mini_train_dit(*_inputs(126, 128))
    assert out.shape == (1, 16, 1, 126, 128)


@torch.no_grad()
def test_compile_blocks_dynamic_seq_allows_ops_peak_probe(tmp_path):
    """op 级峰值探针会读取 shape，dynamic-seq 必须允许 Dynamo 退回静态 guard。"""

    model = _tiny_anima()
    probe = PeakProbe(str(tmp_path / "peak_probe.jsonl"), level="ops", max_steps=1)
    model.enable_peak_probe(probe)
    model.compile_blocks(
        backend="eager",
        dynamic_seq=True,
        bucket_resolutions=[(1008, 1024), (960, 1120)],
    )

    probe.begin_step(0, device=torch.device("cpu"))
    out = model.forward_mini_train_dit(*_inputs(126, 128))
    probe.end_step(device=torch.device("cpu"))

    assert out.shape == (1, 16, 1, 126, 128)
    assert (tmp_path / "peak_probe.jsonl").exists()


@torch.no_grad()
def _run(model: Anima, inp, *, native_flatten: bool) -> torch.Tensor:
    model._native_flatten = native_flatten
    x, timesteps, crossattn_emb = inp
    return model.forward_mini_train_dit(x, timesteps, crossattn_emb)


@torch.no_grad()
def test_flatten_is_bit_exact_4032_family():
    # latent 126x128 → (63)*(64) = 4032 tokens at patch_spatial=2
    model = _tiny_anima()
    inp = _inputs(126, 128)
    out_eager = _run(model, inp, native_flatten=False)
    out_flat = _run(model, inp, native_flatten=True)
    assert torch.equal(out_eager, out_flat)
    assert out_eager.shape == out_flat.shape


@torch.no_grad()
def test_flatten_is_bit_exact_4200_family():
    # latent 120x140 → (60)*(70) = 4200 tokens at patch_spatial=2
    model = _tiny_anima()
    inp = _inputs(120, 140)
    out_eager = _run(model, inp, native_flatten=False)
    out_flat = _run(model, inp, native_flatten=True)
    assert torch.equal(out_eager, out_flat)


@torch.no_grad()
def test_forward_mini_train_dit_derives_and_threads_use_fp32(monkeypatch):
    model = _tiny_anima()
    x, timesteps, crossattn_emb = _inputs(126, 128)
    captured: dict[str, object] = {}
    prepared_x = torch.randn(1, 1, 63, 64, 64, dtype=torch.float16)
    prepared_rope = (
        torch.randn(4032, 1, 1, 16, dtype=torch.float32),
        torch.randn(4032, 1, 1, 16, dtype=torch.float32),
    )

    def fake_run_blocks(self, x_padded, t_embedding_B_T_D, crossattn_emb, attn_params, **block_kwargs):
        captured["run_blocks_use_fp32"] = block_kwargs["use_fp32"]
        captured["rope_dtype"] = (
            None
            if block_kwargs["rope_cos_sin"] is None
            else block_kwargs["rope_cos_sin"][0].dtype
        )
        return x_padded.float() if block_kwargs["use_fp32"] else x_padded

    def fake_final_layer(
        x_B_T_H_W_D,
        emb_B_T_D,
        adaln_lora_B_T_3D=None,
        use_fp32: bool = False,
    ):
        captured["final_layer_use_fp32"] = use_fp32
        return model.final_layer.linear(x_B_T_H_W_D)

    def fake_prepare_embedded_sequence(
        self,
        x_B_C_T_H_W,
        fps=None,
        padding_mask=None,
        h_offset=0,
        w_offset=0,
    ):
        return prepared_x, prepared_rope

    monkeypatch.setattr(
        model,
        "prepare_embedded_sequence",
        types.MethodType(fake_prepare_embedded_sequence, model),
    )
    monkeypatch.setattr(model, "_run_blocks", types.MethodType(fake_run_blocks, model))
    monkeypatch.setattr(model.final_layer, "forward", fake_final_layer)

    out = model.forward_mini_train_dit(x, timesteps, crossattn_emb)

    assert captured["run_blocks_use_fp32"] is True
    assert captured["final_layer_use_fp32"] is True
    assert captured["rope_dtype"] == torch.float32
    assert out.dtype == torch.float32
