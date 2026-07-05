from __future__ import annotations

import torch
from safetensors.torch import save_file

from scripts.merge_to_dit import (
    scan_non_bakeable_adapter,
    scan_non_bakeable_metadata,
)


def test_scan_non_bakeable_metadata_rejects_method_adapters() -> None:
    found = scan_non_bakeable_metadata(
        {
            "ss_network_module": "networks.methods.ip_adapter",
            "ss_network_spec": "ip_adapter",
        }
    )

    assert found == {"IP-Adapter (side network, not a Linear delta)": 1}


def test_scan_non_bakeable_adapter_uses_capability_classifier_for_hydra(tmp_path) -> None:
    adapter = tmp_path / "hydra.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_q_proj.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_q_proj.lora_ups.0.weight": torch.ones(6, 2),
        },
        adapter,
        metadata={"ss_network_spec": "hydra"},
    )

    found = scan_non_bakeable_adapter(
        adapter,
        {
            "lora_unet_blocks_0_q_proj.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_q_proj.lora_ups.0.weight": torch.ones(6, 2),
        },
    )

    assert any("HydraLoRA" in kind for kind in found)
