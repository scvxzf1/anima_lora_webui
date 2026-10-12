"""Compatibility and task safety for all-recorded live metric reads."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web

from web.routes.training import handle_metrics
from web.services.training_service import TrainingService


def test_live_metrics_default_tail_and_explicit_all():
    service = TrainingService.__new__(TrainingService)
    service._metrics_history = [{"step": step, "loss": step} for step in range(7889)]
    assert len(service.get_metrics_history()) == 500
    assert service.get_metrics_history()[0]["step"] == 7389
    assert len(service.get_metrics_history(include_all=True)) == 7889
    assert service.get_metrics_history(include_all=True)[0]["step"] == 0


def test_live_metrics_route_preserves_default_and_task_guard():
    calls = []
    service = SimpleNamespace(
        get_status_snapshot=lambda: {"task_id": "current"},
        get_metrics_history=lambda **kwargs: calls.append(kwargs) or [{"step": 1}],
    )

    def request(query):
        return SimpleNamespace(app={"training_service": service}, query=query)

    response = asyncio.run(handle_metrics(request({"task_id": "current", "all": "1"})))
    assert json.loads(response.text) == [{"step": 1}]
    assert calls == [{"include_all": True}]
    asyncio.run(handle_metrics(request({})))
    assert calls[-1] == {}
    with pytest.raises(web.HTTPConflict):
        asyncio.run(handle_metrics(request({"task_id": "previous", "all": "1"})))
    assert len(calls) == 2
