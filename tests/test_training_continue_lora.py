"""Continue-from-LoRA weight inspection tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: continue_lora

def test_inspect_continue_lora_weight_detects_lora_dora_loha_lokr_and_glora(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    lora_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo.safetensors",
        kind="LoRA",
    )
    dora_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_dora.safetensors",
        kind="DoRA",
    )
    loha_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_loha.safetensors",
        kind="LoHa",
    )
    lokr_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_lokr.safetensors",
        kind="LoKr",
    )
    glora_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_glora.safetensors",
        kind="GLoRA",
    )

    lora_payload = training_service.inspect_continue_lora_weight(
        str(lora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    dora_payload = training_service.inspect_continue_lora_weight(
        str(dora_path),
        variant="dora",
        preset="default",
        methods_subdir="gui-methods",
    )
    dora_blocked = training_service.inspect_continue_lora_weight(
        str(dora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    loha_payload = training_service.inspect_continue_lora_weight(
        str(loha_path),
        variant="loha",
        preset="default",
        methods_subdir="gui-methods",
    )
    loha_blocked = training_service.inspect_continue_lora_weight(
        str(loha_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    lokr_payload = training_service.inspect_continue_lora_weight(
        str(lokr_path),
        variant="lokr",
        preset="default",
        methods_subdir="gui-methods",
    )
    lokr_blocked = training_service.inspect_continue_lora_weight(
        str(lokr_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    glora_payload = training_service.inspect_continue_lora_weight(
        str(glora_path),
        variant="glora",
        preset="default",
        methods_subdir="gui-methods",
    )
    glora_blocked = training_service.inspect_continue_lora_weight(
        str(glora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )

    assert lora_payload["kind"] == "LoRA"
    assert lora_payload["compatible"] is True
    assert dora_payload["kind"] == "DoRA"
    assert dora_payload["compatible"] is True
    assert dora_payload["metadata"]["ss_adapter_variant"] == "dora"
    assert dora_blocked["compatible"] is False
    assert "dora" in dora_blocked["message"].lower()
    assert loha_payload["kind"] == "LoHa"
    assert loha_payload["compatible"] is True
    assert loha_blocked["compatible"] is False
    assert "loha" in loha_blocked["message"].lower()
    assert lokr_payload["kind"] == "LoKr"
    assert lokr_payload["compatible"] is True
    assert lokr_blocked["compatible"] is False
    assert "lokr" in lokr_blocked["message"].lower()
    assert glora_payload["kind"] == "GLoRA"
    assert glora_payload["compatible"] is True
    assert glora_blocked["compatible"] is False
    assert "glora" in glora_blocked["message"].lower()

def test_inspect_continue_lora_weight_rejects_complex_lora_like_weights(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    plain_lora_tensors = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(4, 8),
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(12, 4),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
    }
    cases = [
        (
            "hydra_keys",
            {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.lora_ups.0.weight": torch.randn(12, 4),
                "lora_unet_blocks_0_self_attn_q_proj.router.weight": torch.randn(2, 4),
            },
            None,
        ),
        (
            "stacked_keys",
            {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down_weight": torch.randn(2, 4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight": torch.randn(2, 12, 4),
            },
            None,
        ),
        ("hydra_spec", plain_lora_tensors, {"ss_network_spec": "hydra"}),
        ("stacked_spec", plain_lora_tensors, {"ss_network_spec": "stacked_experts_global_fei"}),
        ("chimera_spec", plain_lora_tensors, {"ss_network_spec": "chimera_hydra"}),
        (
            "reft_key",
            {"reft_unet_blocks_0.rotate_layer.weight": torch.randn(4, 4)},
            {"ss_network_spec": "reft"},
        ),
    ]

    for name, tensors, metadata in cases:
        path = _write_continue_lora_weight(
            tmp_path / "weights" / f"{name}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )
        with pytest.raises(ValueError, match="未识别为 LoRA、DoRA、LoHa、LoKr 或 GLoRA"):
            training_service.inspect_continue_lora_weight(
                str(path),
                variant="lora",
                preset="default",
                methods_subdir="gui-methods",
            )

def test_inspect_continue_lora_weight_reports_path_errors(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="权重文件不存在"):
        training_service.inspect_continue_lora_weight(
            str(tmp_path / "weights" / "missing.safetensors"),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    txt_path = tmp_path / "weights" / "demo.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("not a safetensors file", encoding="utf-8")
    with pytest.raises(ValueError, match="只支持 .safetensors"):
        training_service.inspect_continue_lora_weight(
            str(txt_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    directory_path = tmp_path / "weights" / "directory.safetensors"
    directory_path.mkdir()
    with pytest.raises(ValueError, match="权重路径不是文件"):
        training_service.inspect_continue_lora_weight(
            str(directory_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    unreadable_path = _write_continue_lora_weight(tmp_path / "weights" / "unreadable.safetensors")
    real_access = os.access

    def fake_access(path, mode):
        if Path(path) == unreadable_path and mode == os.R_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(training_service.os, "access", fake_access)
    with pytest.raises(ValueError, match="权重文件不可读取"):
        training_service.inspect_continue_lora_weight(
            str(unreadable_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

