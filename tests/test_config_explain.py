from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from library.config.provenance import explain_key
from scripts.config_explain import build_payload, main


def _write_config_tree(root: Path) -> None:
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
    (root / "methods" / "demo.toml").write_text("network_dim = 32\n", encoding="utf-8")


def _args(**overrides: object) -> argparse.Namespace:
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


def test_build_payload_records_layers_and_overrides(tmp_path: Path) -> None:
    _write_config_tree(tmp_path)

    payload = build_payload(
        _args(
            configs_dir=tmp_path,
            override=[("network_dim", 64)],
        )
    )

    assert payload["values"]["network_dim"] == 64
    assert payload["values"]["blocks_to_swap"] == 12
    assert [item["kind"] for item in explain_key(payload, "network_dim")["history"]] == [
        "base",
        "method",
        "override",
    ]


def test_build_payload_accepts_existing_selected_key(tmp_path: Path) -> None:
    _write_config_tree(tmp_path)

    payload = build_payload(_args(configs_dir=tmp_path, key=["network_dim"]))

    assert explain_key(payload, "network_dim")["value"] == 32


def test_build_payload_rejects_missing_selected_key(tmp_path: Path) -> None:
    _write_config_tree(tmp_path)

    with pytest.raises(KeyError) as exc:
        build_payload(_args(configs_dir=tmp_path, key=["does_not_exist"]))

    assert exc.value.args == ("does_not_exist",)


def test_main_json_selected_key_outputs_only_requested_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_config_tree(tmp_path)
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

    main()

    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == ["network_dim"]
    assert payload["network_dim"]["value"] == 32
