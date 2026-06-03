from __future__ import annotations

import torch

from bench.activation_maps.collector import ActivationCollector
from networks.lora_modules.lora import LoRAModule
from networks.plugins.lokr.module import LoKrModule


class TinyPatchedModel(torch.nn.Module):
    def __init__(self, adapter: torch.nn.Module):
        super().__init__()
        self.linear = adapter.org_module_ref[0]
        self._pgraft_network = torch.nn.Module()
        self._pgraft_network.add_module(adapter.lora_name, adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _lora_model() -> TinyPatchedModel:
    base = torch.nn.Linear(4, 3, bias=False)
    lora = LoRAModule(
        "lora_unet_blocks_0_self_attn_q_proj",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    lora.apply_to()
    with torch.no_grad():
        base.weight.copy_(
            torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, -0.1, 0.1, 0.0],
                    [0.4, 0.3, -0.2, 0.1],
                ]
            )
        )
        lora.lora_down.weight.fill_(0.25)
        lora.lora_up.weight.fill_(0.5)
    return TinyPatchedModel(lora)


def _lokr_model() -> TinyPatchedModel:
    base = torch.nn.Linear(4, 4, bias=False)
    lokr = LoKrModule(
        "lora_unet_blocks_1_cross_attn_q_proj",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.apply_to()
    with torch.no_grad():
        base.weight.copy_(torch.eye(4))
        lokr.lokr_w1.fill_(0.5)
        lokr.lokr_w2.fill_(0.25)
    return TinyPatchedModel(lokr)


def test_activation_collector_records_lora_without_changing_output():
    torch.manual_seed(0)
    model = _lora_model().eval()
    x = torch.randn(2, 4)

    expected = model(x)
    with ActivationCollector(model, record_blocks=False) as collector:
        actual = model(x)

    torch.testing.assert_close(actual, expected)
    assert len(collector.adapter_events) == 1
    event = collector.adapter_events[0]
    assert event.adapter_type == "lora"
    assert event.meta["block_idx"] == 0
    assert event.meta["rank"] == 2
    assert event.delta_to_base > 0
    assert event.bottleneck_summary is not None


def test_activation_collector_records_lokr_delta():
    torch.manual_seed(1)
    model = _lokr_model().eval()
    x = torch.randn(2, 4)

    with ActivationCollector(model, record_blocks=False) as collector:
        _ = model(x)

    assert len(collector.adapter_events) == 1
    event = collector.adapter_events[0]
    assert event.adapter_type == "lokr"
    assert event.meta["block_idx"] == 1
    assert event.meta["factor"] == 2
    assert event.meta["kron_shape"] == [4, 4]
    assert event.bottleneck_summary is None
    assert event.delta_rms > 0
