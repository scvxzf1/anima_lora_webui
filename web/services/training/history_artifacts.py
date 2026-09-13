"""Read-only availability for the existing, path-validated artifact whitelist."""

from .common import HISTORY_ARTIFACT_FILES, HISTORY_RUNTIME_ARTIFACT_FIELDS


def history_artifact_manifest(service, task_id: str) -> dict:
    service.get_history_task_summary(task_id)
    entries = []
    for key in (*HISTORY_ARTIFACT_FILES, *HISTORY_RUNTIME_ARTIFACT_FIELDS):
        entry = {"key": key}
        try:
            path = service.get_history_artifact_path(task_id, key)
            stat = path.stat()
            entry.update(state="available", name=path.name, size_bytes=stat.st_size)
        except FileNotFoundError:
            entry.update(state="missing", message="未保存或文件已不存在")
        except ValueError:
            entry.update(state="blocked", message="文件路径不在允许范围内")
        except OSError:
            entry.update(state="unreadable", message="文件暂不可读")
        entries.append(entry)
    return {"ok": True, "task_id": task_id, "artifacts": entries}
