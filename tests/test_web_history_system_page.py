import json
from web.services.training import history_system


def test_system_page_reaches_prefix_and_final_window(tmp_path, monkeypatch):
    monkeypatch.setattr(history_system, "_history_task_dir", lambda task_id: tmp_path)
    (tmp_path / "system.jsonl").write_text("\n".join(json.dumps({"ts": i, "gpu_util": i}) for i in range(7889)))
    page = history_system.get_history_system_page(None, "task", offset=0)
    assert page["system"][0]["gpu_util"] == 0
    assert page["system"][-1]["gpu_util"] == 1999
    assert page["total"] == 7889 and page["next_offset"] == 2000
    last = history_system.get_history_system_page(None, "task", offset=6000)
    assert last["returned"] == 1889 and last["indices"][-1] == 7888
    assert not last["has_more_after"]


def test_system_page_rejects_invalid_bounds(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(history_system, "_history_task_dir", lambda task_id: tmp_path)
    for options in ({"offset": -1}, {"limit": 0}, {"limit": 2001}):
        with pytest.raises(ValueError):
            history_system.get_history_system_page(None, "task", **options)
