"""Shared training service constants and path helpers."""

from __future__ import annotations

import re
from pathlib import Path

from library.env import get_training_history_root, get_training_queue_root
from web.services._dynamic_path import DynamicPath

# File lives at web/services/training/constants.py → parents[3] is repo root.
# (When ROOT lived in web/services/training_service.py, parents[2] was correct.)
ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = DynamicPath(get_training_history_root)
HISTORY_COLLECTIONS_FILE = DynamicPath(lambda: Path(HISTORY_DIR) / "collections.json")
QUEUE_DIR = DynamicPath(get_training_queue_root)
QUEUE_FILE = DynamicPath(lambda: Path(QUEUE_DIR) / "queue.json")

OUTPUT_READ_SIZE = 4096
MAX_LOG_RECORDS = 3000
MAX_HISTORY_ITEMS = 100
MAX_TIMELINE_LOG_RECORDS = 20000
MAX_TIMELINE_METRIC_RECORDS = 20000
MAX_HISTORY_DETAIL_LOG_RECORDS = 5000
MAX_HISTORY_DETAIL_SYSTEM_RECORDS = 1000
DEFAULT_HISTORY_LOG_PAGE_RECORDS = 360
MAX_HISTORY_LOG_PAGE_RECORDS = 1000
MAX_QUEUE_ITEMS = 200
PROGRESS_RATE_SAMPLE_WINDOW = 9
HISTORY_AVERAGE_SPEED_VERSION = 1
# Live GPU/VRAM sampling for the training dashboard "资源与活动" panel.
# Progress/log lines stream immediately over WS; system metrics used to lag at 5s.
SYSTEM_MONITOR_INTERVAL_SECONDS = 2.0

QUEUE_FAILURE_POLICIES = {"pause", "continue"}
QUEUE_TERMINAL_STATES = {"done", "error", "canceled"}
# 队列批量清理只移除 queue.json 里的列表记录，保留 error 方便确认后重试或手动删除。
QUEUE_CLEARABLE_STATES = {"done", "canceled"}
QUEUE_CLEARABLE_STATE_LABELS = {"done": "已完成", "canceled": "已取消"}

RUNTIME_META_KEYS = (
    "run_dir",
    "runtime_config_file",
    "original_config_file",
    "dataset_config_file",
    "model_cache_dir",
    "dataset_cache_dir",
    "training_output_dir",
    "logs_dir",
    "history_source_config_file",
)

TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?:[^\[]*\[[^\]]*?(?P<rate>[\d.]+)(?P<unit>it/s|s/it)[^\]]*\])?"
)

METRIC_RE = re.compile(
    r"(?:loss[:/]?\s*(?P<loss>[\d.]+))"
    r"|(?:lr[:/]?\s*(?P<lr>[\d.eE\-+]+))"
    r"|(?:norm[:/]?\s*(?P<norm>[\d.]+))"
)


def queue_dir():
    """Return the active queue directory (monkeypatchable via training_service)."""
    from web.services import training_service as facade

    return facade.QUEUE_DIR


def queue_file():
    """Return the active queue file path (monkeypatchable via training_service)."""
    from web.services import training_service as facade

    return facade.QUEUE_FILE


def history_collections_file():
    """Return the active history collections file (monkeypatchable via training_service)."""
    from web.services import training_service as facade

    return facade.HISTORY_COLLECTIONS_FILE


def max_queue_items() -> int:
    """Return the active queue size cap (monkeypatchable via training_service)."""
    from web.services import training_service as facade

    return int(facade.MAX_QUEUE_ITEMS)


def max_history_items() -> int:
    """Return history list cap (monkeypatchable via training_service)."""
    from web.services import training_service as facade

    return int(getattr(facade, "MAX_HISTORY_ITEMS", MAX_HISTORY_ITEMS))


def apply_training_policy_to_facade(policy: dict | None = None) -> dict:
    """Apply durable training policy defaults onto training_service facade constants.

    Capacity/monitor values are applied only when a persisted settings file
    actually defines ``training_policy``. That keeps unit-test monkeypatches
    of ``MAX_QUEUE_ITEMS`` / ``MAX_HISTORY_ITEMS`` stable in default envs.
    """
    from web.services import settings_service, training_service as facade

    settings_file = Path(settings_service.SETTINGS_FILE)
    raw = {}
    if settings_file.exists():
        try:
            import toml
            raw = toml.loads(settings_file.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    has_policy_section = isinstance(raw.get("training_policy"), dict) and bool(raw.get("training_policy"))
    data = policy if isinstance(policy, dict) else settings_service.get_training_policy()
    if has_policy_section or isinstance(policy, dict):
        max_queue = int(data.get("max_queue_items") or MAX_QUEUE_ITEMS)
        max_history = int(data.get("max_history_items") or MAX_HISTORY_ITEMS)
        monitor = float(data.get("system_monitor_interval_sec") or SYSTEM_MONITOR_INTERVAL_SECONDS)
        facade.MAX_QUEUE_ITEMS = max(1, max_queue)
        facade.MAX_HISTORY_ITEMS = max(1, max_history)
        facade.SYSTEM_MONITOR_INTERVAL_SECONDS = max(0.2, monitor)
    return {
        "max_queue_items": int(getattr(facade, "MAX_QUEUE_ITEMS", MAX_QUEUE_ITEMS)),
        "max_history_items": int(getattr(facade, "MAX_HISTORY_ITEMS", MAX_HISTORY_ITEMS)),
        "system_monitor_interval_sec": float(
            getattr(facade, "SYSTEM_MONITOR_INTERVAL_SECONDS", SYSTEM_MONITOR_INTERVAL_SECONDS)
        ),
        "auto_retry": bool(data.get("auto_retry", False)),
        "max_attempts": int(data.get("max_attempts") or 1),
        "retry_backoff_sec": float(data.get("retry_backoff_sec") or 0.0),
    }
