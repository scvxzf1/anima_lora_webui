from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from library.training.gradient_flow_probe import (
    GradientFlowProbe,
    infer_block_group,
)


class _TwoFamilyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_unet_blocks_0_proj = torch.nn.Module()
        self.lora_unet_blocks_0_proj.weight = torch.nn.Parameter(
            torch.tensor([3.0, 4.0])
        )
        self.lora_unet_layers_1_attention_to_q = torch.nn.Module()
        self.lora_unet_layers_1_attention_to_q.weight = torch.nn.Parameter(
            torch.tensor([1.0, 2.0])
        )


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_infer_block_group_accepts_original_and_lora_names() -> None:
    assert infer_block_group("blocks.12.self_attn.qkv_proj.weight") == (
        "blocks.12",
        12,
    )
    assert infer_block_group("lora_unet_blocks_3_mlp_layer2.lora_up.weight") == (
        "blocks.3",
        3,
    )
    assert infer_block_group("layers.29.attention.to_q.weight") == ("layers.29", 29)
    assert infer_block_group("lora_unet_layers_7_attention_to_k.weight") == (
        "layers.7",
        7,
    )
    assert infer_block_group("text_encoder.proj.weight") == ("other", None)


def test_probe_records_exact_gradient_update_and_missing_grad(tmp_path) -> None:
    path = tmp_path / "gradient.jsonl"
    network = _TwoFamilyNetwork()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    block_parameter = network.lora_unet_blocks_0_proj.weight
    block_parameter.grad = torch.tensor([0.3, 0.4])

    probe = GradientFlowProbe(str(path), dense_steps=20, every_n_steps=10)
    probe.capture_before_optimizer(1, network)
    optimizer.step()
    probe.capture_after_optimizer(1)

    event = _events(path)[-1]
    rows = {row["group"]: row for row in event["blocks"]}
    block = rows["blocks.0"]
    layer = rows["layers.1"]

    assert event["step"] == 1
    assert block["parameter_norm"] == pytest.approx(5.0)
    assert block["grad_norm"] == pytest.approx(0.5)
    assert block["relative_grad_norm"] == pytest.approx(0.1)
    assert block["abs_grad_parameter_dot"] == pytest.approx(2.5)
    assert block["elementwise_abs_grad_parameter_sum"] == pytest.approx(2.5)
    assert block["update_norm"] == pytest.approx(0.05)
    assert block["relative_update"] == pytest.approx(0.01)
    assert block["grad_energy_share"] == pytest.approx(1.0)
    assert block["update_energy_share"] == pytest.approx(1.0)
    assert block["first_order_share"] == pytest.approx(1.0)
    assert block["interval_change_norm"] is None
    assert layer["no_grad_tensor_count"] == 1
    assert layer["grad_norm"] == 0.0
    assert layer["update_norm"] == 0.0


def test_probe_records_change_since_previous_sample(tmp_path) -> None:
    path = tmp_path / "gradient.jsonl"
    network = _TwoFamilyNetwork()
    parameter = network.lora_unet_blocks_0_proj.weight
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    probe = GradientFlowProbe(str(path))

    parameter.grad = torch.tensor([0.3, 0.4])
    probe.capture_before_optimizer(1, network)
    optimizer.step()
    probe.capture_after_optimizer(1)

    with torch.no_grad():
        parameter.add_(torch.tensor([0.1, 0.0]))
    parameter.grad = torch.tensor([0.0, 0.0])
    expected_norm = float(parameter.detach().norm())
    probe.capture_before_optimizer(10, network)
    optimizer.step()
    probe.capture_after_optimizer(10)

    event = _events(path)[-1]
    rows = {row["group"]: row for row in event["blocks"]}
    block = rows["blocks.0"]
    assert block["interval_change_norm"] == pytest.approx(0.1)
    assert block["relative_interval_change"] == pytest.approx(0.1 / expected_norm)


def test_interval_change_includes_current_optimizer_update(tmp_path) -> None:
    path = tmp_path / "gradient.jsonl"
    network = _TwoFamilyNetwork()
    parameter = network.lora_unet_blocks_0_proj.weight
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    probe = GradientFlowProbe(str(path))

    parameter.grad = torch.zeros_like(parameter)
    probe.capture_before_optimizer(1, network)
    optimizer.step()
    probe.capture_after_optimizer(1)

    parameter.grad = torch.tensor([1.0, 0.0])
    before_norm = float(parameter.detach().norm())
    probe.capture_before_optimizer(2, network)
    optimizer.step()
    probe.capture_after_optimizer(2)

    rows = {row["group"]: row for row in _events(path)[-1]["blocks"]}
    block = rows["blocks.0"]
    assert block["interval_change_norm"] == pytest.approx(0.1)
    assert block["relative_interval_change"] == pytest.approx(0.1 / before_norm)


def test_probe_cadence_is_one_based_and_keeps_final_step() -> None:
    probe = GradientFlowProbe(
        "/tmp/unused-gradient-flow.jsonl", dense_steps=3, every_n_steps=5
    )

    recorded = [
        step
        for step in range(1, 13)
        if probe.should_record_step(step, max_train_steps=12)
    ]

    assert recorded == [1, 2, 3, 5, 10, 12]


def test_probe_path_resolution_and_main_process_gate(tmp_path) -> None:
    args = SimpleNamespace(
        gradient_flow_probe_jsonl="auto",
        gradient_flow_probe_every_n_steps=7,
        gradient_flow_probe_dense_steps=4,
        output_dir=str(tmp_path / "ckpt"),
        output_name="demo",
    )

    probe = GradientFlowProbe.from_args(args, is_main_process=True)
    assert probe is not None
    assert probe.path == str(tmp_path / "logs" / "demo.gradient_flow.jsonl")
    assert probe.every_n_steps == 7
    assert probe.dense_steps == 4
    assert _events(tmp_path / "logs" / "demo.gradient_flow.jsonl")[0]["ev"] == (
        "gradient_flow_probe_config"
    )
    assert GradientFlowProbe.from_args(args, is_main_process=False) is None

    args.gradient_flow_probe_jsonl = "off"
    assert GradientFlowProbe.from_args(args, is_main_process=True) is None


def test_probe_read_only_unscales_amp_gradients(tmp_path) -> None:
    path = tmp_path / "gradient.jsonl"
    network = _TwoFamilyNetwork()
    parameter = network.lora_unet_blocks_0_proj.weight
    parameter.grad = torch.tensor([30.0, 40.0])
    probe = GradientFlowProbe(str(path))

    probe.capture_before_optimizer(1, network, grad_scale=100.0)
    probe.capture_after_optimizer(1)

    block = _events(path)[-1]["blocks"][0]
    assert block["grad_norm"] == pytest.approx(0.5)
    assert torch.equal(parameter.grad, torch.tensor([30.0, 40.0]))


def test_probe_failure_disables_future_samples_and_releases_snapshots(tmp_path) -> None:
    path = tmp_path / "gradient.jsonl"
    network = _TwoFamilyNetwork()
    probe = GradientFlowProbe(str(path))
    probe.capture_before_optimizer(1, network)

    probe.fail("before_optimizer", 1, RuntimeError("probe failed"))

    assert not probe.should_record_step(2, max_train_steps=2)
    assert probe._pending == []
    assert probe._previous == {}
    error = _events(path)[-1]
    assert error["ev"] == "gradient_flow_probe_error"
    assert error["disabled"] is True
