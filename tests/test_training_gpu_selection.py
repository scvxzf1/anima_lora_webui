from __future__ import annotations

from web.services.training_service import _apply_gpu_whitelist, _normalize_gpu_whitelist


def test_normalize_gpu_whitelist_filters_invalid_and_duplicates():
    assert _normalize_gpu_whitelist(["1", 0, "bad", 1, "-2", 2]) == [1, 0, 2]


def test_apply_gpu_whitelist_sets_cuda_visible_devices():
    env = {}

    _apply_gpu_whitelist(env, [2, 0])

    assert env["CUDA_VISIBLE_DEVICES"] == "2,0"


def test_apply_gpu_whitelist_keeps_default_when_empty():
    env = {"CUDA_VISIBLE_DEVICES": "3"}

    _apply_gpu_whitelist(env, [])

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
