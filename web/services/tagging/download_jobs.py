"""Explicit, cancellable downloads for the optional tagging model assets."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import ipaddress
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from .model_assets import (
    MANIFEST_VERSION,
    ModelAsset,
    ModelAssetFile,
    asset_directory,
    asset_file_path,
    canonical_asset_id,
    inspect_asset,
    iter_model_assets,
    public_asset,
    validate_download_url,
    _huggingface_token,
)

MAX_REDIRECTS = 3
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT = 30.0
DOWNLOAD_SOCKET_TIMEOUT = 120.0
MAX_DOWNLOAD_JOBS = 100
MAX_ERROR_LENGTH = 400

# Clash/sing-box and similar TUN proxies commonly answer DNS with an RFC 2544
# benchmarking address, then intercept that synthetic destination locally.  It
# is safe to admit only this exact IPv4 range here because download URLs remain
# HTTPS-only and host/repository allowlisted.  Ordinary private, loopback and
# link-local addresses are still removed from resolver results.
_PROXY_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


class ModelDownloadError(RuntimeError):
    """A user-facing model download failure with no secret-bearing details."""


@dataclass(slots=True)
class _DownloadJob:
    id: str
    asset_id: str
    total_bytes: int
    total_files: int
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    bytes_downloaded: int = 0
    completed_files: int = 0
    current_file: str = ""
    error: str = ""
    cancel_requested: bool = False
    task: asyncio.Task[Any] | None = None
    stage_dir: Path | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "bytes_downloaded": max(0, self.bytes_downloaded),
            "total_bytes": self.total_bytes,
            "completed_files": self.completed_files,
            "total_files": self.total_files,
            "current_file": self.current_file,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


class _PublicOnlyResolver(AbstractResolver):
    """Resolve download hosts and retain only public or proxy Fake-IP results."""

    def __init__(self, delegate: AbstractResolver | None = None):
        self.delegate = delegate or DefaultResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        results = await self.delegate.resolve(host, port, family)
        allowed = []
        for result in results:
            address = result.get("host", "") if hasattr(result, "get") else getattr(result, "host", "")
            if _is_safe_download_address(address):
                allowed.append(result)
        if not allowed:
            raise OSError("模型下载主机解析到了本机或私有网络地址")
        return allowed

    async def close(self) -> None:
        outcome = self.delegate.close()
        if inspect.isawaitable(outcome):
            await outcome


class ModelDownloadService:
    """Own model asset status and short-lived download jobs.

    The service deliberately keeps jobs in memory.  Downloaded files are the
    durable state; an interrupted process leaves no final files from a staging
    transaction and a later explicit click can safely retry.  This also avoids
    writing transient status or absolute paths into user configuration files.
    """

    def __init__(
        self,
        *,
        assets: Iterable[ModelAsset] | None = None,
        session_factory: Callable[[], Any] | None = None,
        url_factory: Callable[[ModelAsset, ModelAssetFile], str] | None = None,
    ) -> None:
        self._assets = tuple(assets or iter_model_assets())
        self._assets_by_id = {asset.id: asset for asset in self._assets}
        self._session_factory = session_factory
        self._url_factory = url_factory
        self._jobs: dict[str, _DownloadJob] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._download_gate = asyncio.Semaphore(1)
        self._lock = asyncio.Lock()

    async def list_assets(self) -> dict[str, Any]:
        """Return manifest entries and current local status."""

        assets: list[dict[str, Any]] = []
        for asset in self._assets:
            status = await asyncio.to_thread(inspect_asset, asset, verify_hash=False)
            active = self._active_job_for(asset.id)
            if active is not None:
                status = {
                    **status,
                    "state": "downloading",
                    "download_id": active.id,
                    "download": active.snapshot(),
                }
            assets.append(public_asset(asset, status))
        return {
            "ok": True,
            "manifest_version": MANIFEST_VERSION,
            "assets": assets,
            "downloads": self.list_downloads()["downloads"],
        }

    async def get_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self._get_asset(asset_id)
        status = await asyncio.to_thread(inspect_asset, asset, verify_hash=True)
        active = self._active_job_for(asset.id)
        if active is not None:
            status = {
                **status,
                "state": "downloading",
                "download_id": active.id,
                "download": active.snapshot(),
            }
        return {"ok": True, "asset": public_asset(asset, status)}

    async def start_download(self, asset_id: str) -> dict[str, Any]:
        asset = self._get_asset(asset_id)
        active = self._active_job_for(asset.id)
        if active is not None:
            return {
                "ok": True,
                "deduplicated": True,
                "download": active.snapshot(),
            }

        status = await asyncio.to_thread(inspect_asset, asset, verify_hash=True)
        async with self._lock:
            # Re-check under the lock so two simultaneous clicks share a job.
            active = self._active_job_for(asset.id)
            if active is not None:
                return {
                    "ok": True,
                    "deduplicated": True,
                    "download": active.snapshot(),
                }
            if status.get("installed"):
                job = _DownloadJob(
                    id=uuid.uuid4().hex,
                    asset_id=asset.id,
                    total_bytes=asset.total_size,
                    total_files=len(asset.files),
                    state="completed",
                    started_at=time.time(),
                    finished_at=time.time(),
                    bytes_downloaded=asset.total_size,
                    completed_files=len(asset.files),
                )
                self._remember_job(job)
                return {"ok": True, "deduplicated": False, "download": job.snapshot()}
            if asset.requires_auth and not _huggingface_token():
                hint = asset.auth_hint or "请先配置 Hugging Face 凭据后再下载。"
                raise ValueError(hint)

            job = _DownloadJob(
                id=uuid.uuid4().hex,
                asset_id=asset.id,
                total_bytes=asset.total_size,
                total_files=len(asset.files),
            )
            self._remember_job(job)
            task = asyncio.create_task(self._run(job, asset), name=f"tagging-download-{job.id}")
            job.task = task
            self._tasks.add(task)
            task.add_done_callback(self._task_done)
            return {"ok": True, "deduplicated": False, "download": job.snapshot()}

    def list_downloads(self) -> dict[str, Any]:
        jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return {"ok": True, "downloads": [job.snapshot() for job in jobs]}

    def get_download(self, download_id: str) -> dict[str, Any]:
        job = self._jobs.get(str(download_id or "").strip())
        if job is None:
            raise KeyError("模型下载任务不存在")
        return {"ok": True, "download": job.snapshot()}

    async def cancel_download(self, download_id: str) -> dict[str, Any]:
        job = self._jobs.get(str(download_id or "").strip())
        if job is None:
            raise KeyError("模型下载任务不存在")
        if job.state in {"completed", "error", "canceled"}:
            return {"ok": True, "download": job.snapshot()}
        job.cancel_requested = True
        job.state = "cancel_requested"
        task = job.task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if job.state not in {"canceled", "error", "completed"}:
            job.state = "canceled"
            job.finished_at = time.time()
        return {"ok": True, "download": job.snapshot()}

    async def shutdown(self) -> None:
        tasks = []
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.cancel_requested = True
                job.task.cancel()
                tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run(self, job: _DownloadJob, asset: ModelAsset) -> None:
        stage_dir = asset_directory(asset) / f".download-{job.id}"
        job.stage_dir = stage_dir
        session = None
        try:
            async with self._download_gate:
                self._check_cancel(job)
                job.state = "downloading"
                job.started_at = time.time()
                status = await asyncio.to_thread(inspect_asset, asset, verify_hash=True)
                valid_files = {
                    str(item.get("path")): bool(item.get("valid"))
                    for item in status.get("files", [])
                    if isinstance(item, dict)
                }
                for item in status.get("files", []):
                    if isinstance(item, dict) and item.get("valid"):
                        job.bytes_downloaded += int(item.get("size", 0) or 0)
                        job.completed_files += 1
                await asyncio.to_thread(_prepare_stage, stage_dir)
                session = await self._create_session()
                for declared in asset.files:
                    self._check_cancel(job)
                    if valid_files.get(declared.path):
                        continue
                    job.current_file = declared.path
                    stage_path = _stage_file_path(stage_dir, declared)
                    part_path = Path(f"{stage_path}.part")
                    await asyncio.to_thread(_ensure_parent, stage_path)
                    await self._download_file(session, asset, declared, part_path, job)
                    await asyncio.to_thread(os.replace, part_path, stage_path)
                    job.completed_files += 1
                job.state = "publishing"
                for declared in asset.files:
                    self._check_cancel(job)
                    if valid_files.get(declared.path):
                        continue
                    job.current_file = declared.path
                    stage_path = _stage_file_path(stage_dir, declared)
                    target = asset_file_path(asset, declared)
                    await asyncio.to_thread(_ensure_parent, target)
                    await asyncio.to_thread(os.replace, stage_path, target)
                final_status = await asyncio.to_thread(inspect_asset, asset, verify_hash=True)
                if not final_status.get("installed"):
                    raise ModelDownloadError("模型文件发布后完整性校验失败")
                job.bytes_downloaded = asset.total_size
                job.completed_files = len(asset.files)
                job.current_file = ""
                job.state = "completed"
                job.finished_at = time.time()
        except asyncio.CancelledError:
            job.state = "canceled"
            job.finished_at = time.time()
            raise
        except Exception as exc:  # noqa: BLE001 - surface a bounded user error
            job.state = "error"
            job.error = _safe_error(exc)
            job.finished_at = time.time()
        finally:
            if session is not None:
                await self._close_session(session)
            job.current_file = "" if job.state in {"completed", "error", "canceled"} else job.current_file
            await asyncio.to_thread(_cleanup_stage, stage_dir)

    async def _download_file(
        self,
        session: Any,
        asset: ModelAsset,
        declared: ModelAssetFile,
        part_path: Path,
        job: _DownloadJob,
    ) -> None:
        raw_url = self._url_factory(asset, declared) if self._url_factory else asset.url_for(declared)
        url = validate_download_url(raw_url)
        response = await self._request_with_redirects(session, url)
        try:
            status = int(getattr(response, "status", 0) or 0)
            if status != 200:
                detail = await _read_response_error(response)
                raise ModelDownloadError(f"模型服务器返回 HTTP {status}{(': ' + detail) if detail else ''}")
            content_length = _header_int(response, "Content-Length")
            if content_length is not None and content_length != declared.size:
                raise ModelDownloadError(f"文件大小与 manifest 不符：{declared.path}")
            digest = hashlib.sha256()
            received = 0
            await asyncio.to_thread(_ensure_parent, part_path)
            with part_path.open("wb") as handle:
                while True:
                    self._check_cancel(job)
                    chunk = await _response_read(response, DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > declared.size:
                        raise ModelDownloadError(f"文件超过 manifest 声明大小：{declared.path}")
                    digest.update(chunk)
                    await asyncio.to_thread(handle.write, chunk)
                    job.bytes_downloaded += len(chunk)
                await asyncio.to_thread(_flush_file, handle)
            if received != declared.size or digest.hexdigest() != declared.sha256:
                raise ModelDownloadError(f"文件完整性校验失败：{declared.path}")
        except asyncio.CancelledError:
            raise
        finally:
            _release_response(response)

    async def _request_with_redirects(self, session: Any, url: str) -> Any:
        current = validate_download_url(url)
        for _index in range(MAX_REDIRECTS + 1):
            validate_download_url(current)
            try:
                result = session.get(current, allow_redirects=False)
                response = await result if inspect.isawaitable(result) else result
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, OSError) as exc:
                raise ModelDownloadError("无法连接模型服务器：" + _safe_error(exc)) from exc
            status = int(getattr(response, "status", 0) or 0)
            if 300 <= status < 400:
                location = _response_header(response, "Location")
                _release_response(response)
                if not location:
                    raise ModelDownloadError("模型服务器返回了无目标的重定向")
                current = validate_download_url(urljoin(current, location))
                continue
            return response
        raise ModelDownloadError("模型服务器重定向次数过多")

    async def _create_session(self) -> Any:
        if self._session_factory is not None:
            value = self._session_factory()
            return await value if inspect.isawaitable(value) else value
        resolver = _PublicOnlyResolver()
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=DOWNLOAD_CONNECT_TIMEOUT,
            sock_connect=DOWNLOAD_CONNECT_TIMEOUT,
            sock_read=DOWNLOAD_SOCKET_TIMEOUT,
        )
        headers = {"Accept": "application/octet-stream"}
        token = _huggingface_token()
        if token:
            # The token is kept in memory and attached only to the HTTPS HF
            # session. It is never included in job snapshots or error text.
            headers["Authorization"] = f"Bearer {token}"
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False, limit=2),
            headers=headers,
        )

    async def _close_session(self, session: Any) -> None:
        close = getattr(session, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _get_asset(self, asset_id: str) -> ModelAsset:
        key = str(asset_id or "").strip()
        asset = self._assets_by_id.get(key)
        if asset is None:
            # Profiles created by older versions may still carry a repository
            # ID or variant key.  Resolve those aliases at the service edge,
            # while keeping injected/custom manifests exact-match only.
            asset = self._assets_by_id.get(canonical_asset_id(key))
        if asset is None:
            # Keep the canonical error for callers using the default manifest;
            # custom test manifests still get the same fail-closed behavior.
            raise KeyError(f"模型资产不存在：{key or '空值'}")
        return asset

    def _active_job_for(self, asset_id: str) -> _DownloadJob | None:
        for job in self._jobs.values():
            if job.asset_id == asset_id and job.state in {"queued", "downloading", "publishing", "cancel_requested"}:
                return job
        return None

    def _remember_job(self, job: _DownloadJob) -> None:
        self._jobs[job.id] = job
        if len(self._jobs) <= MAX_DOWNLOAD_JOBS:
            return
        finished = sorted(
            (item for item in self._jobs.values() if item.state in {"completed", "error", "canceled"}),
            key=lambda item: item.created_at,
        )
        for old in finished[: max(0, len(self._jobs) - MAX_DOWNLOAD_JOBS)]:
            self._jobs.pop(old.id, None)

    def _check_cancel(self, job: _DownloadJob) -> None:
        if job.cancel_requested:
            raise asyncio.CancelledError()

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        # A task canceled before its coroutine got a chance to run still needs
        # a terminal state for polling clients.
        for job in self._jobs.values():
            if job.task is task and job.state in {"queued", "cancel_requested"}:
                job.state = "canceled"
                job.finished_at = time.time()
                break
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass


def _prepare_stage(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _cleanup_stage(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except OSError:
        pass


def _stage_file_path(stage_dir: Path, declared: ModelAssetFile) -> Path:
    relative = Path(declared.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelDownloadError("manifest 文件路径越界")
    path = (stage_dir / relative).resolve()
    try:
        path.relative_to(stage_dir.resolve())
    except ValueError as exc:
        raise ModelDownloadError("manifest 文件路径越界") from exc
    return path


def _ensure_parent(path: Path) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    # Refuse a pre-existing symlink anywhere in the controlled parent chain.
    current = parent
    while current.exists():
        if current.is_symlink():
            raise ModelDownloadError("模型目录包含不安全的符号链接")
        if current.parent == current:
            break
        current = current.parent


def _flush_file(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


async def _response_read(response: Any, size: int) -> bytes:
    content = getattr(response, "content", None)
    if content is not None and hasattr(content, "read"):
        value = content.read(size)
    else:
        value = response.read(size)
    return await value if inspect.isawaitable(value) else value


async def _read_response_error(response: Any) -> str:
    try:
        raw = await _response_read(response, 2048)
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(raw, bytes):
        return " ".join(raw.decode("utf-8", errors="replace").split())[:MAX_ERROR_LENGTH]
    return _safe_error(raw)


def _header_int(response: Any, key: str) -> int | None:
    raw = _response_header(response, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _response_header(response: Any, key: str) -> str | None:
    headers = getattr(response, "headers", {}) or {}
    try:
        value = headers.get(key)
    except AttributeError:
        value = None
    return str(value) if value is not None else None


def _release_response(response: Any) -> None:
    release = getattr(response, "release", None)
    if callable(release):
        release()
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _safe_error(exc: Any) -> str:
    return " ".join(str(exc or "").split())[:MAX_ERROR_LENGTH] or "未知错误"


def _is_private_address(value: Any) -> bool:
    try:
        address = ipaddress.ip_address(str(value or "").strip())
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _is_safe_download_address(value: Any) -> bool:
    try:
        address = ipaddress.ip_address(str(value or "").strip())
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address) and address in _PROXY_FAKE_IP_NETWORK:
        return True
    return bool(
        address.is_global
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_reserved
        and not address.is_multicast
        and not address.is_unspecified
    )


__all__ = ["MAX_DOWNLOAD_JOBS", "ModelDownloadError", "ModelDownloadService"]
