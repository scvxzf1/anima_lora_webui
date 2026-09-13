import pytest

from scripts.experiments import dynamic_block_swap_comparison as comparison


def test_comparison_rejects_incomplete_run(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "summarize", lambda _: {"acceptance": {"passed": False}})
    with pytest.raises(ValueError, match="accepted complete"):
        comparison.compare(tmp_path, tmp_path)


def test_comparison_keeps_loss_and_descriptive_timing_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "summarize", lambda _: {"acceptance": {"passed": True}})
    monkeypatch.setattr(comparison, "measured_updates", lambda path: [{
        "step": step, "shapes": [[1, 16, 8, 8]],
        "seconds": 1 if path.name == "dynamic" else 2,
        "loss": 1 if path.name == "dynamic" else 1.01,
    } for step in range(1, 97)])
    result = comparison.compare(tmp_path / "dynamic", tmp_path / "fixed")
    assert result["same_step_loss"]["max_abs"] == pytest.approx(0.01)
    assert not result["causal_speedup_proven"]
    assert list(result["dynamic_after_paired_verdict_66"].values())[0]["n"] == 30


def test_comparison_rejects_mismatched_shapes(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "summarize", lambda _: {"acceptance": {"passed": True}})
    monkeypatch.setattr(comparison, "measured_updates", lambda path: [{
        "step": 1, "shapes": [[1, 16, 8, 8 if path.name == "dynamic" else 16]],
        "seconds": 1, "loss": 1,
    }])
    with pytest.raises(ValueError, match="Step/shape mismatch"):
        comparison.compare(tmp_path / "dynamic", tmp_path / "fixed")
