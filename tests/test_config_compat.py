from __future__ import annotations

import argparse
from pathlib import Path

from scripts.config_compat import build_payload


def _write_config_tree(root: Path) -> None:
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
    (root / "presets.toml").write_text("[default]\nblocks_to_swap = 0\n", encoding="utf-8")
    (root / "methods" / "demo.toml").write_text("network_dim = 32\n", encoding="utf-8")


def _args(**overrides: object) -> argparse.Namespace:
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


def test_build_payload_reports_compile_mutation_for_method_trace(tmp_path: Path) -> None:
    _write_config_tree(tmp_path)

    payload = build_payload(
        _args(
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


def test_build_payload_accepts_direct_config_file(tmp_path: Path) -> None:
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

    payload = build_payload(_args(config_file=str(config)))

    assert payload["ok"] is False
    assert payload["source"] == str(config)
    assert "block_swap_soft_tokens" in _codes(payload["errors"])


def test_build_payload_applies_overrides_to_direct_config_file(tmp_path: Path) -> None:
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

    payload = build_payload(_args(config_file=str(config), override=[("blocks_to_swap", 0)]))

    assert payload["ok"] is True
    assert payload["effective"]["blocks_to_swap"] == 0
    assert "block_swap_soft_tokens" not in _codes(payload["errors"])
