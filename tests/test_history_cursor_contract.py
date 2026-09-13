"""Pagination tests use synthetic summaries only, never a live history directory."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from web.routes.training_history_list import history_list_response, history_page


def test_cursor_survives_new_tasks_deletion_and_tied_timestamps():
    tasks = [{"id": name, "started_at": 10} for name in ["a", "d", "c", "b"]]
    first = history_page(tasks, limit=2)
    assert [item["id"] for item in first["tasks"]] == ["d", "c"]
    changed = [task for task in tasks if task["id"] != "c"]
    changed.append({"id": "new", "started_at": 20})
    second = history_page(changed, limit=2, cursor=first["next_cursor"])
    assert [item["id"] for item in second["tasks"]] == ["b", "a"]
    assert second["next_cursor"] is None
    assert second["total"] == 4


def test_total_is_not_truncated_by_storage_default_limit():
    tasks = [{"id": str(i), "started_at": i} for i in range(350)]
    service = Mock()
    service.list_history_tasks.side_effect = lambda **kw: tasks if kw["limit"] == 0 else tasks[:100]
    request = SimpleNamespace(app={"training_service": service}, query={"limit": "200"})
    response = asyncio.run(history_list_response(request))
    result = json.loads(response.text)
    assert response.status == 200
    assert len(result["tasks"]) == 200
    assert result["total"] == 350
    assert result["next_cursor"].startswith("v1.")
    service.list_history_tasks.assert_called_once_with(include_archived=False, limit=0)


def test_search_is_forwarded_before_paging():
    service = Mock()
    service.list_history_tasks.return_value = [{"id": "older"}]
    request = SimpleNamespace(app={"training_service": service}, query={"q": " older ", "limit": "200"})
    result = json.loads(asyncio.run(history_list_response(request)).text)
    assert result["search"] == "older"
    assert result["total"] == 1
    service.list_history_tasks.assert_called_once_with(include_archived=False, limit=0, search="older")


@pytest.mark.parametrize("cursor", ["-1", "invalid", "v1.!!", "v1.bm9uZQ", "v1.W3RydWUsImEiXQ"])
def test_invalid_cursor_fails_closed(cursor):
    with pytest.raises(ValueError, match="游标"):
        history_page([], limit=200, cursor=cursor)


@pytest.mark.parametrize("limit", ["bad", "-1", "0", "100001"])
def test_invalid_limit_returns_400_without_loading_history(limit):
    service = Mock()
    request = SimpleNamespace(app={"training_service": service}, query={"limit": limit})
    assert asyncio.run(history_list_response(request)).status == 400
    service.list_history_tasks.assert_not_called()


def test_initial_numeric_offset_compatibility():
    result = history_page([{"id": str(i), "started_at": i} for i in range(5)], limit=2, cursor="2")
    assert [item["id"] for item in result["tasks"]] == ["2", "1"]
    assert result["next_cursor"].startswith("v1.")
