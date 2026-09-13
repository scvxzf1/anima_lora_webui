from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import toml

from scripts.experiments.family_auto_swap_hot import prepare_fixture
from library.training.auto_block_swap.config import configuration_errors


@pytest.mark.parametrize("family", ["anima", "z_image"])
def test_fixture_copies_correct_family_caches_and_bounds_training(tmp_path, family):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (896, 1200)).save(source / "real.png")
    (source / "real.txt").write_text("caption")
    (source / f"real_0896x1200_{family}.npz").write_bytes(b"latent fixture")
    (source / f"real_{family}_te.safetensors").write_bytes(b"text fixture")
    config = source / "source.toml"
    config.write_text('sample_at_first = true\ntorch_compile = true\n'
                      'validate_every_n_steps = 0\nvalidate_every_n_epochs = 0\n')
    before = {p: p.read_bytes() for p in source.iterdir()}
    options = SimpleNamespace(
        output=tmp_path / "run", images=source, cache=source, config=config,
        family=family, dit=Path("model"), qwen3=Path("text"), vae=Path("vae"),
        limit=4, steps=16, reserve_percent=25.0, preference="vram",
    )
    prepare_fixture(options)
    saved = toml.load(options.output / "config.toml")
    assert saved["model_family"] == family
    assert saved["auto_block_swap_mode"] == "startup"
    assert saved["auto_block_swap_vram_reserve_percent"] == 25
    assert saved["auto_block_swap_preference"] == "vram"
    assert saved["max_train_steps"] == 16
    assert not saved["torch_compile"] and not saved["sample_at_first"]
    assert saved["max_data_loader_n_workers"] == 0
    assert saved.get("validate_every_n_steps") is None
    assert saved.get("validate_every_n_epochs") is None
    assert not configuration_errors(saved)
    assert {p: p.read_bytes() for p in source.iterdir()} == before
    assert len(list((options.output / "cache").iterdir())) == 2
    with pytest.raises(FileExistsError):
        prepare_fixture(options)


@pytest.mark.parametrize("family", ["anima", "z_image"])
def test_hot_acceptance_does_not_claim_krea_only_features(family):
    for config in ({"auto_block_swap_mode": "dynamic"}, {"auto_block_swap_preference": "ram"}):
        errors = configuration_errors({"auto_block_swap": True, "model_family": family, **config})
        assert any("Krea-2" in error for error in errors)


def test_report_rejects_gpu_success_with_host_paging(tmp_path):
    from library.training.auto_block_swap.process import write_result
    from scripts.experiments.family_auto_swap_report import summarize

    directory = tmp_path / "calibration"
    directory.mkdir()
    write_result(directory / "summary.json", {"status": "calibrating", "trials": [{
        "blocks": 28, "status": "ok", "safe": True, "seconds": 7,
        "host_min_available": 100, "host_reserve": 10, "swap_io_bytes": 128 * 1024**2,
    }]})
    result = summarize(tmp_path)
    assert not result["valid"]
    assert not result["trials"][0]["accepted_resource_probe"]
    assert "paging" in result["trials"][0]["rejection"]


@pytest.mark.parametrize("field,value,error", [
    ("host_available", 10, "host reserve"),
    ("swap_io", 128 * 1024**2, "paging budget"),
    ("headroom", 1, "GPU reserve"),
    ("loss", float("inf"), "Nonfinite loss"),
])
def test_report_checks_recorded_updates_independently(tmp_path, field, value, error):
    import json
    from library.training.auto_block_swap.process import write_result
    from scripts.experiments.family_auto_swap_report import summarize

    (tmp_path / "calibration").mkdir()
    (tmp_path / "formal").mkdir()
    probe = {"status": "ok", "safe": True, "seconds": 1, "headroom": 100,
             "host_min_available": 100, "host_reserve": 10, "swap_io_bytes": 0}
    write_result(tmp_path / "calibration/summary.json", {
        "status": "selected", "selected_blocks": 26, "confirmation": probe,
    })
    write_result(tmp_path / "formal/completion.json", {"status": "completed", "steps": 1})
    write_result(tmp_path / "acceptance.json", {"status": "passed", "gpu_reserve": 10})
    update = {"step": 1, "blocks": 26, "seconds": 1, "loss": 0.1,
              "host_available": 100, "swap_io": 0, "headroom": 100}
    path = tmp_path / "hot-updates.json"
    write_result(path, {"updates": [update]})
    assert summarize(tmp_path)["valid"]
    update[field] = value
    path.write_text(json.dumps({"updates": [update]}))
    assert any(error in item for item in summarize(tmp_path)["errors"])
