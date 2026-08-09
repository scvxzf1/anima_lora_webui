from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from library.config.provenance import explain_key
from scripts.config_compat import build_payload as build_compat_payload
from scripts.config_explain import (
    build_payload as build_explain_payload,
    main as explain_main,
)


def _write_compat_config_tree(root: Path) -> None:
    (root / "methods").mkdir(parents=True)
    (root / "base.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                "blocks_to_swap = 0",
                "gradient_checkpointing = false",
                "torch_compile = true",
                'dynamo_backend = "inductor"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "presets.toml").write_text(
        "[default]\nblocks_to_swap = 0\n",
        encoding="utf-8",
    )
    (root / "methods" / "demo.toml").write_text(
        "network_dim = 32\n",
        encoding="utf-8",
    )


def _compat_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        config_file=None,
        method="demo",
        preset="default",
        methods_subdir="methods",
        configs_dir="configs",
        runtime_config=None,
        override=[],
        json=False,
        strict=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _codes(items: list[dict]) -> set[str]:
    return {item["code"] for item in items}


def test_compat_reports_compile_mutation_for_method_trace(tmp_path: Path) -> None:
    _write_compat_config_tree(tmp_path)

    payload = build_compat_payload(
        _compat_args(
            configs_dir=tmp_path,
            override=[
                ("blocks_to_swap", 8),
                ("dynamo_backend", "cudagraphs"),
            ],
        )
    )

    assert payload["ok"] is True
    assert payload["effective"]["blocks_to_swap"] == 8
    assert "block_swap_cudagraphs_disable_compile" in _codes(payload["warnings"])
    assert payload["mutations"] == [
        {
            "code": "block_swap_cudagraphs_disable_compile",
            "key": "torch_compile",
            "value": False,
            "message": (
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile."
            ),
        }
    ]


def test_compat_accepts_direct_config_file(tmp_path: Path) -> None:
    config = tmp_path / "config.runtime.toml"
    config.write_text(
        "\n".join(
            [
                'network_module = "networks.methods.soft_tokens"',
                "blocks_to_swap = 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_compat_payload(_compat_args(config_file=str(config)))

    assert payload["ok"] is False
    assert payload["source"] == str(config)
    assert "block_swap_soft_tokens" in _codes(payload["errors"])


def test_compat_applies_overrides_to_direct_config_file(tmp_path: Path) -> None:
    config = tmp_path / "config.runtime.toml"
    config.write_text(
        "\n".join(
            [
                'network_module = "networks.methods.soft_tokens"',
                "blocks_to_swap = 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_compat_payload(
        _compat_args(config_file=str(config), override=[("blocks_to_swap", 0)])
    )

    assert payload["ok"] is True
    assert payload["effective"]["blocks_to_swap"] == 0
    assert "block_swap_soft_tokens" not in _codes(payload["errors"])


def _write_explain_config_tree(root: Path) -> None:
    (root / "methods").mkdir(parents=True)
    (root / "base.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                "network_dim = 8",
                "blocks_to_swap = 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "presets.toml").write_text(
        "\n".join(
            [
                "[default]",
                "blocks_to_swap = 0",
                "[low]",
                "blocks_to_swap = 12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "methods" / "demo.toml").write_text(
        "network_dim = 32\n",
        encoding="utf-8",
    )


def _explain_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        method="demo",
        preset="low",
        methods_subdir="methods",
        configs_dir="configs",
        runtime_config=None,
        key=None,
        override=[],
        json=False,
        strict=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_explain_records_layers_and_overrides(tmp_path: Path) -> None:
    _write_explain_config_tree(tmp_path)

    payload = build_explain_payload(
        _explain_args(
            configs_dir=tmp_path,
            override=[("network_dim", 64)],
        )
    )

    assert payload["values"]["network_dim"] == 64
    assert payload["values"]["blocks_to_swap"] == 12
    assert [
        item["kind"] for item in explain_key(payload, "network_dim")["history"]
    ] == [
        "base",
        "method",
        "override",
    ]


def test_explain_accepts_existing_selected_key(tmp_path: Path) -> None:
    _write_explain_config_tree(tmp_path)

    payload = build_explain_payload(
        _explain_args(configs_dir=tmp_path, key=["network_dim"])
    )

    assert explain_key(payload, "network_dim")["value"] == 32


def test_explain_rejects_missing_selected_key(tmp_path: Path) -> None:
    _write_explain_config_tree(tmp_path)

    with pytest.raises(KeyError) as exc:
        build_explain_payload(
            _explain_args(configs_dir=tmp_path, key=["does_not_exist"])
        )

    assert exc.value.args == ("does_not_exist",)


def test_explain_main_json_outputs_only_requested_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_explain_config_tree(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "config_explain.py",
            "--method",
            "demo",
            "--preset",
            "low",
            "--configs-dir",
            str(tmp_path),
            "--key",
            "network_dim",
            "--json",
        ],
    )

    explain_main()

    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == ["network_dim"]
    assert payload["network_dim"]["value"] == 32
