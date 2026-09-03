"""Cancelable OpenAI-compatible vision client used by tagging jobs."""

from __future__ import annotations

import asyncio
import base64
import inspect
import ipaddress
import json
import mimetypes
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from .settings import get_api_key, validate_endpoint

MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class TaggingApiError(RuntimeError):
    """An external provider failure with enough information for retry policy."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        attempts: int = 1,
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.attempts = attempts


class _PublicOnlyResolver(AbstractResolver):
    """Resolve through aiohttp, rejecting addresses the connector would use."""

    def __init__(self, *, allow_private: bool, delegate: AbstractResolver | None = None):
        self.allow_private = allow_private
        self.delegate = delegate or DefaultResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        results = await self.delegate.resolve(host, port, family)
        if self.allow_private:
            return results
        for result in results:
            address = result.get("host", "") if hasattr(result, "get") else getattr(result, "host", "")
            if _is_private_ip_or_name(address):
                raise OSError("API 主机解析到了本机或私有网络地址")
        return results

    async def close(self) -> None:
        outcome = self.delegate.close()
        if inspect.isawaitable(outcome):
            await outcome


class OpenAICompatibleClient:
    """OpenAI-compatible JSON client with cancelable requests and retries."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.base_url = validate_endpoint(settings).rstrip("/")
        # A job may be bound to a non-active profile.  Preserve the legacy
        # zero-argument call when no profile id is present so embedded callers
        # that replace ``get_api_key`` continue to work unchanged.
        profile_id = settings.get("_profile_id")
        self.api_key = get_api_key(profile_id) if profile_id else get_api_key()

    async def describe_image(self, image_path: Path, prompt: str) -> dict[str, Any]:
        path = Path(image_path)
        size = await asyncio.to_thread(lambda: path.stat().st_size)
        if size > MAX_IMAGE_BYTES:
            raise TaggingApiError(
                f"图片超过外部 API 上传上限（{MAX_IMAGE_BYTES // (1024 * 1024)} MiB）",
                retryable=False,
            )
        data = await asyncio.to_thread(path.read_bytes)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(data).decode("ascii")
        body = {
            "model": self.settings.get("model") or "",
            "messages": [
                {"role": "system", "content": self.settings.get("system_prompt") or ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
            "temperature": 0.2,
        }
        started = time.perf_counter()
        response, attempts = await self._request_with_retries("/chat/completions", body)
        caption = extract_caption(response)
        if not caption:
            raise TaggingApiError("模型返回了空的 caption", retryable=False, attempts=attempts)
        return {
            "caption": caption,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": response.get("model") or self.settings.get("model") or "",
            "attempts": attempts,
        }

    async def ping(self) -> dict[str, Any]:
        started = time.perf_counter()
        response, _attempts = await self._request_with_retries("/models", None, retries=0)
        models = response.get("data") if isinstance(response, dict) else []
        model_ids = [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]
        selected = str(self.settings.get("model") or "")
        return {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model_available": not selected or selected in model_ids,
            "models": model_ids[:100],
        }

    async def actual(self) -> dict[str, Any]:
        """Make a cheap text-only completion to verify auth and model routing."""

        started = time.perf_counter()
        body = {
            "model": self.settings.get("model") or "",
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 8,
            "temperature": 0,
        }
        response, _attempts = await self._request_with_retries("/chat/completions", body)
        text = extract_caption(response)
        return {"ok": True, "elapsed_ms": round((time.perf_counter() - started) * 1000), "response": text[:200]}

    async def _request_with_retries(
        self,
        path: str,
        body: dict[str, Any] | None,
        *,
        retries: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        max_retries = int(self.settings.get("retry_count", 2) if retries is None else retries)
        interval = float(self.settings.get("retry_interval_seconds", 1.5) or 0)
        for attempt in range(max_retries + 1):
            try:
                return await self._request_once(path, body), attempt + 1
            except TaggingApiError as exc:
                exc.attempts = attempt + 1
                if not exc.retryable or attempt >= max_retries:
                    raise
                await asyncio.sleep(interval * (2**attempt))
        raise TaggingApiError("外部 API 请求失败", retryable=False, attempts=max_retries + 1)

    async def _request_once(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        url = _join_url(self.base_url, path)
        _validate_literal_target(url, allow_private=bool(self.settings.get("allow_private_network")))
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = aiohttp.ClientTimeout(total=float(self.settings.get("timeout_seconds", 120) or 120))
        resolver = _PublicOnlyResolver(allow_private=bool(self.settings.get("allow_private_network")))
        connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                try:
                    async with session.request(
                        "POST" if body is not None else "GET",
                        url,
                        json=body,
                        headers=headers,
                        allow_redirects=False,
                    ) as response:
                        raw = await response.content.read(MAX_RESPONSE_BYTES + 1)
                        status = response.status
                except asyncio.TimeoutError as exc:
                    raise TaggingApiError("外部 API 请求超时", retryable=True) from exc
                except (aiohttp.ClientError, OSError) as exc:
                    raise TaggingApiError("无法连接外部 API：" + _safe_error_text(exc), retryable=True) from exc
        finally:
            await resolver.close()

        if len(raw) > MAX_RESPONSE_BYTES:
            raise TaggingApiError("外部 API 响应体过大", retryable=False)
        if 300 <= status < 400:
            raise TaggingApiError(f"外部 API 返回 HTTP {status} 重定向；为保护 API Key 已拒绝跟随", status=status)
        if status < 200 or status >= 300:
            detail = _read_error_body(raw)
            raise TaggingApiError(
                f"外部 API 返回 HTTP {status}{(': ' + detail) if detail else ''}",
                status=status,
                retryable=status == 429 or status >= 500,
            )
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TaggingApiError("外部 API 返回了非 JSON 响应", retryable=False) from exc
        if not isinstance(value, dict):
            raise TaggingApiError("外部 API 返回格式不是 JSON object", retryable=False)
        return value


def extract_caption(payload: dict[str, Any]) -> str:
    """Extract text from common OpenAI-compatible response variants."""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        value = message.get("content") or choice.get("text")
        text = _content_to_text(value)
        if text:
            return _clean_caption(text)
    return _clean_caption(_content_to_text(payload.get("output")))


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                part = item.get("text") or item.get("content")
                if isinstance(part, str):
                    parts.append(part)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        return _content_to_text(value.get("text") or value.get("content"))
    return ""


def _clean_caption(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    return text[:100000]


def _join_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    base_path = parsed.path.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return parsed._replace(path=f"{base_path}{suffix}").geturl()


def _read_error_body(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
        if isinstance(payload, dict):
            message = payload.get("error")
            if isinstance(message, dict):
                message = message.get("message")
            if message:
                return _safe_error_text(message)
    except json.JSONDecodeError:
        pass
    return _safe_error_text(raw.decode("utf-8", errors="replace"))


def _safe_error_text(value: Any) -> str:
    return " ".join(str(value or "").split())[:300]


def _validate_literal_target(url: str, *, allow_private: bool) -> None:
    if allow_private:
        return
    hostname = urlsplit(url).hostname or ""
    if _is_private_ip_or_name(hostname):
        raise TaggingApiError("API URL 指向本机或私有网络；如确认安全，请开启允许私有网络 API")


def _is_private_ip_or_name(value: str) -> bool:
    host = str(value or "").strip().lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
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
