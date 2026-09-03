"""Optional local Danbooru tag translation dictionary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from library.env import anima_home
from web.services.atomic_io import atomic_write_text

from .download_jobs import _PublicOnlyResolver

SOURCE_NAME = "ffdkj/Danbooru Tag Chinese-English Translation Table"
SOURCE_COMMIT = "bc2953723a76e1841e9564297c6812723223ecb0"
SOURCE_URL = (
    "https://raw.githubusercontent.com/ffdkj/"
    "ffdkj-Danbooru_Tag-Chinese-English-Translation-Table/"
    f"{SOURCE_COMMIT}/tag.sqlite"
)
SOURCE_SIZE = 23_937_024
SOURCE_SHA256 = "08671c8ca1f0342baa5e9e6cfd8ab64d9a703165a2c4dbd7212c6115680ef1a8"
MAX_ENTRIES = 200_000
MAX_TAGS_PER_REQUEST = 500
ROOT_ENV = "ANIMA_TAG_DICTIONARY_ROOT"
TERMINAL_STATES = {"completed", "error", "canceled"}
_UNSET_STAMP = object()


@dataclass(slots=True)
class _DownloadJob:
    id: str
    state: str = "queued"
    bytes_downloaded: int = 0
    total_bytes: int = SOURCE_SIZE
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str = ""
    task: asyncio.Task[Any] | None = None
    cancel_requested: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class TagDictionaryService:
    """Download, cache, and query a fixed-version translation table."""

    def __init__(self) -> None:
        self._jobs: dict[str, _DownloadJob] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._download_lock = asyncio.Lock()
        self._cache_lock = threading.Lock()
        self._entries: dict[str, str] | None = None
        self._reverse: dict[str, str] | None = None
        self._cache_stamp = 0
        self._meta_cache_stamp: object | tuple[int, int, int, int, int, int] | None = _UNSET_STAMP
        self._meta_cache: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        meta = self._installed_meta()
        active = self._active_job()
        return {
            "ok": True,
            "installed": bool(meta),
            "state": "downloading" if active else ("installed" if meta else "missing"),
            "source_name": SOURCE_NAME,
            "source_commit": SOURCE_COMMIT,
            "download_size": SOURCE_SIZE,
            "entry_count": int((meta or {}).get("entry_count", 0) or 0),
            "installed_at": int((meta or {}).get("installed_at", 0) or 0),
            **({"download": active.snapshot()} if active else {}),
        }

    async def start_download(self) -> dict[str, Any]:
        async with self._download_lock:
            active = self._active_job()
            if active:
                return {"ok": True, "deduplicated": True, "download": active.snapshot()}
            if self._installed_meta():
                now = time.time()
                job = _DownloadJob(
                    id=uuid.uuid4().hex,
                    state="completed",
                    bytes_downloaded=SOURCE_SIZE,
                    started_at=now,
                    finished_at=now,
                )
                self._remember(job)
                return {"ok": True, "deduplicated": False, "download": job.snapshot()}
            job = _DownloadJob(id=uuid.uuid4().hex)
            self._remember(job)
            task = asyncio.create_task(self._run_download(job), name=f"tag-dictionary-{job.id}")
            job.task = task
            self._tasks.add(task)
            task.add_done_callback(self._task_done)
            return {"ok": True, "deduplicated": False, "download": job.snapshot()}

    def get_download(self, download_id: str) -> dict[str, Any]:
        job = self._jobs.get(str(download_id or "").strip())
        if job is None:
            raise KeyError("标签词典下载任务不存在")
        return {"ok": True, "download": job.snapshot()}

    async def cancel_download(self, download_id: str) -> dict[str, Any]:
        job = self._jobs.get(str(download_id or "").strip())
        if job is None:
            raise KeyError("标签词典下载任务不存在")
        if job.state in TERMINAL_STATES:
            return {"ok": True, "download": job.snapshot()}
        job.cancel_requested = True
        if job.task and not job.task.done():
            if job.state in {"queued", "downloading"}:
                job.state = "cancel_requested"
                job.task.cancel()
            elif job.state == "building":
                job.state = "cancel_requested"
            await asyncio.gather(job.task, return_exceptions=True)
        if job.state not in TERMINAL_STATES:
            job.state = "canceled"
            job.finished_at = time.time()
        return {"ok": True, "download": job.snapshot()}

    async def translate(self, tags: list[Any], target_language: str) -> dict[str, Any]:
        target = str(target_language or "").strip().lower()
        if target not in {"zh", "en"}:
            raise ValueError("target_language 只支持 zh 或 en")
        if not isinstance(tags, list) or not tags:
            raise ValueError("tags 必须是非空数组")
        if len(tags) > MAX_TAGS_PER_REQUEST:
            raise ValueError(f"单次最多翻译 {MAX_TAGS_PER_REQUEST} 个 tag")
        clean = [str(tag or "").strip()[:500] for tag in tags]
        if any(not tag for tag in clean):
            raise ValueError("tag 不能为空")
        return await asyncio.to_thread(self._translate_sync, clean, target)

    async def shutdown(self) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for job in self._jobs.values():
            task = job.task
            if task is None or task.done():
                continue
            job.cancel_requested = True
            if job.state in {"queued", "downloading"}:
                job.state = "cancel_requested"
                task.cancel()
            elif job.state == "building":
                job.state = "cancel_requested"
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _translate_sync(self, tags: list[str], target: str) -> dict[str, Any]:
        entries, reverse = self._load_indexes()
        translated: list[str] = []
        matched = 0
        for tag in tags:
            if target == "zh":
                result = entries.get(_normalize_english(tag), tag)
            else:
                result = reverse.get(_normalize_chinese(tag), tag)
            matched += int(result != tag)
            translated.append(result)
        return {
            "ok": True,
            "target_language": target,
            "translations": translated,
            "matched": matched,
            "total": len(tags),
        }

    def _load_indexes(self) -> tuple[dict[str, str], dict[str, str]]:
        active = _active_file()
        try:
            stamp = active.stat().st_mtime_ns
        except OSError as exc:
            raise RuntimeError("请先下载本地中英标签词典") from exc
        with self._cache_lock:
            if self._entries is not None and self._reverse is not None and self._cache_stamp == stamp:
                return self._entries, self._reverse
            try:
                payload = json.loads(active.read_text(encoding="utf-8"))
                raw_entries = payload.get("entries") if isinstance(payload, dict) else None
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("本地标签词典损坏，请重新下载") from exc
            if not isinstance(raw_entries, dict) or not raw_entries:
                raise RuntimeError("本地标签词典为空，请重新下载")
            entries = {
                _normalize_english(key): str(value).strip()
                for key, value in raw_entries.items()
                if str(key).strip() and str(value).strip()
            }
            reverse: dict[str, str] = {}
            for english, chinese in entries.items():
                reverse.setdefault(_normalize_chinese(chinese), english)
            self._entries = entries
            self._reverse = reverse
            self._cache_stamp = stamp
            return entries, reverse

    async def _run_download(self, job: _DownloadJob) -> None:
        stage: Path | None = None
        session: aiohttp.ClientSession | None = None
        try:
            root = tag_dictionary_root()
            stage = root / f".download-{job.id}.sqlite"
            root.mkdir(parents=True, exist_ok=True)
            self._check_cancel(job)
            job.state = "downloading"
            job.started_at = time.time()
            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=120)
            session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(resolver=_PublicOnlyResolver(), use_dns_cache=False, limit=1),
            )
            async with session.get(SOURCE_URL, allow_redirects=False) as response:
                if response.status != 200:
                    raise RuntimeError(f"词典服务器返回 HTTP {response.status}")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) != SOURCE_SIZE:
                    raise RuntimeError("标签词典大小与固定 manifest 不符")
                digest = hashlib.sha256()
                received = 0
                with stage.open("wb") as handle:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        received += len(chunk)
                        if received > SOURCE_SIZE:
                            raise RuntimeError("标签词典超过固定 manifest 大小")
                        digest.update(chunk)
                        handle.write(chunk)
                        job.bytes_downloaded = received
                    handle.flush()
                    os.fsync(handle.fileno())
            if received != SOURCE_SIZE or digest.hexdigest() != SOURCE_SHA256:
                raise RuntimeError("标签词典完整性校验失败")
            self._check_cancel(job)
            job.state = "building"
            parse_task = asyncio.create_task(asyncio.to_thread(_parse_sqlite, stage))
            try:
                entries = await asyncio.shield(parse_task)
            except asyncio.CancelledError:
                await asyncio.gather(parse_task, return_exceptions=True)
                raise
            self._check_cancel(job)
            if not entries:
                raise RuntimeError("标签词典解析结果为空")
            meta = {
                "source_name": SOURCE_NAME,
                "source_commit": SOURCE_COMMIT,
                "source_sha256": SOURCE_SHA256,
                "entry_count": len(entries),
                "installed_at": int(time.time()),
            }
            job.state = "publishing"
            publish_task = asyncio.create_task(asyncio.to_thread(_publish_dictionary, stage, entries, meta))
            try:
                await asyncio.shield(publish_task)
            except asyncio.CancelledError:
                outcome = await asyncio.gather(publish_task, return_exceptions=True)
                if outcome and isinstance(outcome[0], BaseException):
                    raise RuntimeError(str(outcome[0]) or "标签词典发布失败") from outcome[0]
                # Publishing is the commit point. Once it starts, report the
                # durable result instead of claiming that installation stopped.
            with self._cache_lock:
                self._entries = None
                self._reverse = None
                self._cache_stamp = 0
            job.bytes_downloaded = SOURCE_SIZE
            job.state = "completed"
            job.finished_at = time.time()
        except asyncio.CancelledError:
            job.state = "canceled"
            job.finished_at = time.time()
            raise
        except Exception as exc:  # noqa: BLE001 - present a bounded download error
            job.state = "error"
            job.error = str(exc or "标签词典下载失败")[:400]
            job.finished_at = time.time()
        finally:
            if session is not None:
                await session.close()
            try:
                if stage is not None:
                    stage.unlink(missing_ok=True)
            except OSError:
                pass

    def _active_job(self) -> _DownloadJob | None:
        return next((job for job in self._jobs.values() if job.state not in TERMINAL_STATES), None)

    def _installed_meta(self) -> dict[str, Any] | None:
        stamp = _installation_stamp()
        if stamp == self._meta_cache_stamp:
            return self._meta_cache
        self._meta_cache = _read_meta()
        self._meta_cache_stamp = stamp
        return self._meta_cache

    def _remember(self, job: _DownloadJob) -> None:
        self._jobs[job.id] = job
        if len(self._jobs) > 20:
            finished = sorted((item for item in self._jobs.values() if item.state in TERMINAL_STATES), key=lambda item: item.created_at)
            for old in finished[: len(self._jobs) - 20]:
                self._jobs.pop(old.id, None)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        for job in self._jobs.values():
            if job.task is task and job.state not in TERMINAL_STATES:
                job.state = "canceled"
                job.finished_at = time.time()
                break
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _check_cancel(job: _DownloadJob) -> None:
        if job.cancel_requested:
            raise asyncio.CancelledError()


def tag_dictionary_root() -> Path:
    raw = os.environ.get(ROOT_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
        if ".." in path.parts:
            raise ValueError(f"{ROOT_ENV} 不能包含 ..")
        if not path.is_absolute():
            path = anima_home() / path
        return path.resolve()
    return (anima_home() / "models" / "tag-dictionaries" / "danbooru-zh-en").resolve()


def _parse_sqlite(path: Path) -> dict[str, str]:
    uri = f"{Path(path).resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT name, cn_name FROM tags ORDER BY post_count DESC LIMIT ?",
            (MAX_ENTRIES,),
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, str] = {}
    for name, chinese in rows:
        english = _normalize_english(name)
        translated = str(chinese or "").strip()
        if english and translated:
            result[english] = translated
    return result


def _publish_dictionary(stage: Path, entries: dict[str, str], meta: dict[str, Any]) -> None:
    root = tag_dictionary_root()
    root.mkdir(parents=True, exist_ok=True)
    # meta.json is the final commit marker read by status().
    os.replace(stage, _source_file())
    atomic_write_text(_active_file(), json.dumps({"meta": meta, "entries": entries}, ensure_ascii=False, separators=(",", ":")))
    atomic_write_text(_meta_file(), json.dumps(meta, ensure_ascii=False, separators=(",", ":")))


def _read_meta() -> dict[str, Any] | None:
    try:
        value = json.loads(_meta_file().read_text(encoding="utf-8"))
        active_payload = json.loads(_active_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or not isinstance(active_payload, dict):
        return None
    if value.get("source_commit") != SOURCE_COMMIT or value.get("source_sha256") != SOURCE_SHA256:
        return None
    try:
        entry_count = int(value.get("entry_count", 0) or 0)
        source_size = _source_file().stat().st_size
    except (OSError, TypeError, ValueError):
        return None
    active_meta = active_payload.get("meta")
    entries = active_payload.get("entries")
    if not isinstance(active_meta, dict) or not isinstance(entries, dict):
        return None
    if not 0 < entry_count <= MAX_ENTRIES or source_size != SOURCE_SIZE or len(entries) != entry_count:
        return None
    for key in ("source_commit", "source_sha256", "entry_count"):
        if active_meta.get(key) != value.get(key):
            return None
    if not entries:
        return None
    return value


def _installation_stamp() -> tuple[int, int, int, int, int, int] | None:
    try:
        meta = _meta_file().stat()
        active = _active_file().stat()
        source = _source_file().stat()
    except OSError:
        return None
    return (
        meta.st_mtime_ns,
        meta.st_size,
        active.st_mtime_ns,
        active.st_size,
        source.st_mtime_ns,
        source.st_size,
    )


def _normalize_english(value: Any) -> str:
    return " ".join(str(value or "").strip().replace("_", " ").casefold().split())


def _normalize_chinese(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _active_file() -> Path:
    return tag_dictionary_root() / "active.json"


def _meta_file() -> Path:
    return tag_dictionary_root() / "meta.json"


def _source_file() -> Path:
    return tag_dictionary_root() / "source.sqlite"


__all__ = ["TagDictionaryService", "tag_dictionary_root"]
