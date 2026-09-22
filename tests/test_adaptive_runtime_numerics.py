from library.training.adaptive_runtime.numerics import discover_finite_plan


def failure(name="projection"):
    return {"status": "error", "failure_kind": "nonfinite", "first_nonfinite": {
        "module": name, "type": "Linear", "inputs": [{"finite": True}],
        "outputs": [{"finite": False}],
    }}


def test_promotion_remains_finite_only_not_calibrated():
    calls = []

    def runner(promoted, **kwargs):
        calls.append(promoted)
        return failure() if not promoted else {"status": "ok"}

    report = discover_finite_plan(runner, record=lambda report: None)
    assert calls == [(), ("projection",)]
    assert report["status"] == "finite_only"
    assert report["precision_calibrated"] is False


def test_repeated_failure_stops_and_oom_does_not_promote():
    report = discover_finite_plan(lambda *a, **k: failure(), record=lambda report: None)
    assert len(report["attempts"]) == 2
    report = discover_finite_plan(lambda *a, **k: {**failure(), "status": "cuda_oom"},
                                  record=lambda report: None)
    assert len(report["attempts"]) == 1
    assert report["fp32_modules"] == []


def test_nonfinite_input_is_not_a_promotion_candidate():
    result = failure()
    result["first_nonfinite"]["inputs"][0]["finite"] = False
    report = discover_finite_plan(lambda *a, **k: result, record=lambda report: None)
    assert report["status"] == "failed" and report["fp32_modules"] == []


def test_invalid_worker_structures_stop_without_promoting():
    for result in (None, {"status": "error", "first_nonfinite": []},
                   {"status": "error", "first_nonfinite": {"inputs": [None]}}):
        report = discover_finite_plan(lambda *a, **k: result, record=lambda report: None)
        assert report["status"] == "failed" and len(report["attempts"]) == 1


def test_attempt_limit_does_not_publish_untested_promotion():
    report = discover_finite_plan(lambda *a, **k: failure(), max_attempts=1,
                                  record=lambda report: None)
    assert report["fp32_modules"] == []


def test_stale_nonfinite_trace_does_not_hide_runtime_failure():
    result = {**failure(), "failure_kind": "runtime", "error": "dtype mismatch"}
    report = discover_finite_plan(lambda *a, **k: result, record=lambda report: None)
    assert len(report["attempts"]) == 1 and report["fp32_modules"] == []
