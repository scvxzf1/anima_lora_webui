from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save, save_file

from web.services import preview_service, settings_service, weight_analysis_service


def _patch_weight_analysis_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    settings_file = root / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(
        '\n'.join(
            [
                '[global]',
                'output_root = "output/runs"',
                '',
                '[preview]',
                'training_dir = "output/ckpt/sample"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(weight_analysis_service, "ROOT", root)
    monkeypatch.setattr(preview_service, "ROOT", root)
    monkeypatch.setattr(preview_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(settings_service, "ROOT", root)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return root


def _training_output(root: Path) -> Path:
    out = root / "output" / "runs" / "001-demo" / "training_output"
    out.mkdir(parents=True)
    return out


def test_lora_delta_norm_alpha_rank_and_block_parse(tmp_path, monkeypatch):
    root = _patch_weight_analysis_root(tmp_path, monkeypatch)
    out = _training_output(root)
    path = out / "demo.safetensors"
    prefix = "lora_unet_blocks_13_mlp_layer1"
    down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    save_file(
        {
            f"{prefix}.lora_down.weight": down,
            f"{prefix}.lora_up.weight": up,
            f"{prefix}.alpha": torch.tensor(1.0),
        },
        str(path),
        metadata={"ss_output_name": "demo", "ss_network_spec": "lora"},
    )

    payload = weight_analysis_service.inspect_weight(str(path))
    expected = (up @ down) * 0.5

    assert payload["ok"] is True
    assert payload["adapter_type"] == "LoRA"
    assert payload["unsupported"]["unsupported"] is False
    assert payload["summary"]["layer_count"] == 1
    layer = payload["layers"][0]
    assert layer["name"] == prefix
    assert layer["block"] == 13
    assert layer["component"] == "mlp_layer1"
    assert layer["rank"] == 2
    assert layer["alpha"] == 1.0
    assert math.isclose(layer["fro_norm"], float(torch.linalg.vector_norm(expected)), rel_tol=1e-6)
    assert math.isclose(layer["mean_abs"], float(expected.abs().mean()), rel_tol=1e-6)
    assert math.isclose(layer["max_abs"], float(expected.abs().max()), rel_tol=1e-6)
    assert payload["component_summary"][0]["label"] == "mlp_layer1"
    assert payload["block_summary"][0]["block"] == 13
    assert payload["style_top20"][0]["name"] == prefix
    assert payload["heatmap"]["blocks"] == [13]
    assert "mlp_layer1" in payload["heatmap"]["components"]

    file_uri_payload = weight_analysis_service.inspect_weight(path.resolve().as_uri())
    assert file_uri_payload["summary"]["layer_count"] == 1


def test_loha_and_lokr_minimal_weights_are_supported(tmp_path, monkeypatch):
    root = _patch_weight_analysis_root(tmp_path, monkeypatch)
    out = _training_output(root)

    loha = out / "loha.safetensors"
    loha_prefix = "lora_unet_blocks_2_self_attn_q_proj"
    save_file(
        {
            f"{loha_prefix}.hada_w1_a": torch.tensor([[1.0], [2.0]]),
            f"{loha_prefix}.hada_w1_b": torch.tensor([[3.0, 4.0]]),
            f"{loha_prefix}.hada_w2_a": torch.tensor([[1.0], [1.0]]),
            f"{loha_prefix}.hada_w2_b": torch.tensor([[1.0, 2.0]]),
            f"{loha_prefix}.alpha": torch.tensor(1.0),
        },
        str(loha),
        metadata={"ss_network_spec": "loha"},
    )
    loha_payload = weight_analysis_service.inspect_weight(str(loha))
    assert loha_payload["adapter_type"] == "LoHa"
    assert loha_payload["summary"]["layer_count"] == 1
    assert loha_payload["layers"][0]["component"] == "self_attn_q_proj"

    lokr = out / "lokr.safetensors"
    lokr_prefix = "lora_unet_blocks_3_cross_attn_k_proj"
    save_file(
        {
            f"{lokr_prefix}.lokr_w1": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            f"{lokr_prefix}.lokr_w2": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            f"{lokr_prefix}.alpha": torch.tensor(2.0),
        },
        str(lokr),
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "2"},
    )
    lokr_payload = weight_analysis_service.inspect_weight(str(lokr))
    assert lokr_payload["adapter_type"] == "LoKr"
    assert lokr_payload["summary"]["layer_count"] == 1
    assert lokr_payload["layers"][0]["component"] == "cross_attn_k_proj"
    assert lokr_payload["layers"][0]["rank"] == 2


def test_invalid_missing_and_escaped_paths_are_rejected(tmp_path, monkeypatch):
    root = _patch_weight_analysis_root(tmp_path, monkeypatch)
    out = _training_output(root)
    bad_ext = out / "demo.ckpt"
    bad_ext.write_bytes(b"not safetensors")

    with pytest.raises(ValueError, match="只支持 .safetensors"):
        weight_analysis_service.inspect_weight(str(bad_ext))

    with pytest.raises(FileNotFoundError):
        weight_analysis_service.inspect_weight(str(out / "missing.safetensors"))

    outside = tmp_path / "outside.safetensors"
    save_file({"x": torch.zeros(1)}, str(outside))
    with pytest.raises(ValueError, match="训练输出目录或全局输出目录"):
        weight_analysis_service.inspect_weight(str(outside))

    with pytest.raises(ValueError, match="不能包含"):
        weight_analysis_service.inspect_weight("../outside.safetensors")


def test_unsupported_adapter_returns_stable_json_shape(tmp_path, monkeypatch):
    root = _patch_weight_analysis_root(tmp_path, monkeypatch)
    out = _training_output(root)
    path = out / "vera.safetensors"
    save_file(
        {"lora_unet_blocks_0_self_attn_q_proj.vera_lambda_b": torch.ones(2)},
        str(path),
        metadata={"ss_network_spec": "vera"},
    )

    payload = weight_analysis_service.inspect_weight(str(path))

    assert payload["ok"] is True
    assert payload["adapter_type"] == "VeRA"
    assert payload["unsupported"]["unsupported"] is True
    assert payload["layers"] == []
    assert payload["component_summary"] == []
    assert payload["block_summary"] == []
    assert payload["style_top20"] == []
    assert payload["character_top20"] == []
    assert payload["heatmap"] == {"blocks": [], "components": [], "matrix": [], "max_value": 0.0, "cells": []}


def test_dora_weight_analysis_is_marked_unsupported_without_base_weight(tmp_path, monkeypatch):
    root = _patch_weight_analysis_root(tmp_path, monkeypatch)
    out = _training_output(root)
    path = out / "dora.safetensors"
    prefix = "lora_unet_blocks_0_self_attn_q_proj"
    save_file(
        {
            f"{prefix}.lora_down.weight": torch.ones(1, 2),
            f"{prefix}.lora_up.weight": torch.ones(3, 1),
            f"{prefix}.alpha": torch.tensor(1.0),
            f"{prefix}.dora_scale": torch.ones(3),
        },
        str(path),
        metadata={"ss_network_spec": "dora", "ss_adapter_variant": "dora"},
    )

    payload = weight_analysis_service.inspect_weight(str(path))

    assert payload["adapter_type"] == "DoRA"
    assert payload["unsupported"]["unsupported"] is True
    assert "底模权重" in payload["unsupported"]["reason"]
    assert payload["layers"] == []


def test_uploaded_safetensors_bytes_are_inspected_without_path_boundary(tmp_path, monkeypatch):
    _patch_weight_analysis_root(tmp_path, monkeypatch)
    prefix = "lora_unet_blocks_18_cross_attn_v_proj"
    down = torch.tensor([[1.0, 2.0]])
    up = torch.tensor([[3.0], [4.0]])
    data = save(
        {
            f"{prefix}.lora_down.weight": down,
            f"{prefix}.lora_up.weight": up,
            f"{prefix}.alpha": torch.tensor(1.0),
        },
        metadata={"ss_network_spec": "lora", "ss_output_name": "upload"},
    )

    payload = weight_analysis_service.inspect_weight_bytes(data, filename="dragged.safetensors")

    assert payload["ok"] is True
    assert payload["file"]["source"] == "upload"
    assert payload["file"]["path"] == "uploaded://dragged.safetensors"
    assert payload["metadata"]["ss_output_name"] == "upload"
    assert payload["adapter_type"] == "LoRA"
    assert payload["summary"]["layer_count"] == 1
    assert payload["layers"][0]["block"] == 18
    assert payload["layers"][0]["component"] == "cross_attn_v_proj"
    assert "临时读取" in payload["disclaimer"]


def test_list_analysis_weights_reuses_training_weight_listing(tmp_path, monkeypatch):
    root = _patch_weight_analysis_root(tmp_path, monkeypatch)
    out = _training_output(root)
    path = out / "listed.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_mlp_layer1.lora_down.weight": torch.ones(1, 2),
            "lora_unet_blocks_0_mlp_layer1.lora_up.weight": torch.ones(2, 1),
        },
        str(path),
        metadata={"ss_output_name": "listed"},
    )

    payload = weight_analysis_service.list_analysis_weights(
        task={"id": "task-demo", "job": "training", "output_dir": str(out), "variant": "listed"},
        allow_latest_fallback=False,
    )

    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["weights"][0]["name"] == "listed.safetensors"
    assert payload["weights"][0]["abs_path"] == str(path.resolve())
    assert "不加载模型" in payload["analysis_note"]
