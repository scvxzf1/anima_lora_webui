from __future__ import annotations

from pathlib import Path

from library.config.provenance import explain_key, trace_method_config


def _write_config_tree(root: Path) -> None:
    (root / "methods").mkdir(parents=True)
    (root / "base.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                "network_dim = 8",
                "blocks_to_swap = 0",
                "learning_rate = 0.0001",
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
                "learning_rate = 0.0002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "methods" / "demo.toml").write_text(
        "\n".join(
            [
                "network_dim = 32",
                "learning_rate = 0.0003",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_trace_method_config_records_layer_history(tmp_path: Path) -> None:
    _write_config_tree(tmp_path)

    trace = trace_method_config("demo", "low", configs_dir=tmp_path)

    assert trace["values"]["network_dim"] == 32
    assert trace["values"]["blocks_to_swap"] == 12
    assert trace["values"]["learning_rate"] == 0.0003
    learning_rate = explain_key(trace, "learning_rate")
    assert [item["kind"] for item in learning_rate["history"]] == [
        "base",
        "preset",
        "method",
    ]
    assert learning_rate["source"].endswith("configs/methods/demo.toml") or learning_rate[
        "source"
    ].endswith("/methods/demo.toml")


def test_trace_method_config_layers_runtime_and_overrides(tmp_path: Path) -> None:
    _write_config_tree(tmp_path)
    runtime = tmp_path / "runtime.toml"
    runtime.write_text("network_dim = 48\nblocks_to_swap = 20\n", encoding="utf-8")

    trace = trace_method_config(
        "demo",
        "low",
        configs_dir=tmp_path,
        runtime_config=runtime,
        overrides={"network_dim": 64},
    )

    assert trace["values"]["network_dim"] == 64
    assert trace["values"]["blocks_to_swap"] == 20
    network_dim = explain_key(trace, "network_dim")
    assert [item["kind"] for item in network_dim["history"]] == [
        "base",
        "method",
        "runtime",
        "override",
    ]
    assert network_dim["source"] == "CLI/override"
