from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.tagging import setup_tagging_routes
from web.services.tagging import download_jobs
from web.services.tagging.download_jobs import ModelDownloadService
from web.services.tagging.download_jobs import _PublicOnlyResolver
from web.services.tagging.model_assets import (
    MODEL_ASSETS,
    ModelAsset,
    ModelAssetFile,
    asset_directory,
    asset_file_path,
    canonical_asset_id,
    get_model_asset,
    inspect_asset,
    public_asset,
    validate_download_url,
)


def _asset(data: bytes = b"fixture-model") -> tuple[ModelAsset, bytes]:
    declared = ModelAssetFile(
        path="model.onnx",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return (
        ModelAsset(
            id="fixture-asset",
            provider="wd14",
            label="Fixture asset",
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
            revision="0" * 40,
            files=(declared,),
            model_path="model.onnx",
        ),
        data,
    )


class _Content:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self, _size: int) -> bytes:
        data, self._data = self._data, b""
        return data


class _Response:
    status = 200

    def __init__(self, data: bytes):
        self.headers = {"Content-Length": str(len(data))}
        self.content = _Content(data)
        self.released = False

    def release(self) -> None:
        self.released = True


class _Session:
    def __init__(self, data: bytes):
        self.data = data
        self.responses: list[_Response] = []
        self.closed = False

    async def get(self, _url: str, **_kwargs):
        response = _Response(self.data)
        self.responses.append(response)
        return response

    async def close(self) -> None:
        self.closed = True


class _GateContent:
    def __init__(self, data: bytes):
        self._data = data
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def read(self, _size: int) -> bytes:
        self.started.set()
        await self.release.wait()
        data, self._data = self._data, b""
        return data


class _GateResponse(_Response):
    def __init__(self, content: _GateContent, size: int):
        self.headers = {"Content-Length": str(size)}
        self.content = content
        self.released = False


class _GateSession:
    def __init__(self, data: bytes):
        self.content = _GateContent(data)
        self.closed = False

    async def get(self, _url: str, **_kwargs):
        return _GateResponse(self.content, len(self.content._data))

    async def close(self) -> None:
        self.closed = True


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, data: bytes = b"fixture-model"):
    monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
    asset, payload = _asset(data)
    session = _Session(payload)
    service = ModelDownloadService(
        assets=[asset],
        session_factory=lambda: session,
        url_factory=lambda _asset, _file: "https://huggingface.co/fixture/file",
    )
    return service, asset, payload, session


def test_manifest_contains_all_target_local_tagger_presets_without_installing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
    assert [asset.id for asset in MODEL_ASSETS] == [
        "wd14-eva02-large-v3",
        "wd14-vit-v3",
        "wd14-vit-large-v3",
        "wd14-convnext-v2",
        "cltagger-v1-02",
        "cltagger-v2-01a",
    ]
    assert [asset.repo_id for asset in MODEL_ASSETS[:4]] == [
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    ]
    v2 = get_model_asset("cltagger-v2-01a")
    assert v2.requires_auth is True
    assert [item.path for item in v2.files] == [
        "v2_01a/model.onnx",
        "v2_01a/model.onnx.data",
        "v2_01a/model_vocabulary.json",
    ]
    assert canonical_asset_id("SmilingWolf/wd-vit-tagger-v3") == "wd14-vit-v3"
    assert canonical_asset_id("cl_tagger_v2_v2_01a") == "cltagger-v2-01a"
    assert not (tmp_path / "models").exists()


def test_public_asset_exposes_repo_id_without_local_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
    asset = get_model_asset("wd14-vit-v3")
    public = public_asset(asset)
    assert public["repo_id"] == asset.repo_id
    assert public["repo"] == asset.repo_id
    assert str(tmp_path) not in str(public)


def test_gated_download_requires_credentials_before_creating_job(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
        asset, _payload = _asset()
        gated = replace(
            asset,
            requires_auth=True,
            auth_hint="请先登录 Hugging Face。",
        )
        monkeypatch.setattr(
            "web.services.tagging.download_jobs._huggingface_token", lambda: ""
        )
        service = ModelDownloadService(assets=[gated])
        with pytest.raises(ValueError, match="请先登录 Hugging Face"):
            await service.start_download(gated.id)
        assert service.list_downloads()["downloads"] == []
        assert not (tmp_path / "models").exists()
        await service.shutdown()

    asyncio.run(run())


def test_default_download_service_resolves_target_repository_alias() -> None:
    service = ModelDownloadService()
    assert service._get_asset("SmilingWolf/wd-vit-tagger-v3").id == "wd14-vit-v3"


def test_default_download_session_keeps_hf_token_in_request_headers_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Session:
        async def close(self) -> None:
            return None

    def fake_client_session(**kwargs):
        captured.update(kwargs)
        return _Session()

    monkeypatch.setattr(download_jobs, "_huggingface_token", lambda: "fixture-token")
    monkeypatch.setattr(download_jobs.aiohttp, "ClientSession", fake_client_session)

    async def run() -> None:
        service = ModelDownloadService()
        session = await service._create_session()
        assert captured["headers"] == {
            "Accept": "application/octet-stream",
            "Authorization": "Bearer fixture-token",
        }
        assert "fixture-token" not in str(service.list_downloads())
        await service._close_session(session)

    asyncio.run(run())


async def _wait_terminal(service: ModelDownloadService, download_id: str) -> dict:
    for _ in range(100):
        snapshot = service.get_download(download_id)["download"]
        if snapshot["state"] in {"completed", "error", "canceled"}:
            return snapshot
        await asyncio.sleep(0.005)
    raise AssertionError("download fixture did not reach a terminal state")


def test_manifest_paths_and_status_are_hash_checked(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
    asset, payload = _asset()
    assert inspect_asset(asset)["state"] == "missing"

    target = asset_file_path(asset, asset.files[0])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong")
    assert inspect_asset(asset)["state"] == "corrupt"

    target.write_bytes(payload)
    checked = inspect_asset(asset)
    assert checked["state"] == "installed"
    assert checked["files"][0]["verified"] is True
    assert asset_directory(asset) == tmp_path / "models" / "wd14" / asset.id


def test_manifest_rejects_traversal_and_non_https_downloads() -> None:
    with pytest.raises(ValueError):
        ModelAssetFile("../model.onnx", 1, "0" * 64)
    with pytest.raises(ValueError):
        validate_download_url("http://huggingface.co/model")
    with pytest.raises(ValueError):
        validate_download_url("https://example.invalid/model")
    with pytest.raises(ValueError):
        validate_download_url("https://huggingface.co:444/model")


def test_download_resolver_supports_proxy_fake_ip_without_allowing_private_hosts() -> None:
    class _Resolver:
        def __init__(self, addresses: list[str]):
            self.addresses = addresses

        async def resolve(self, host, port=0, family=0):
            return [
                {
                    "hostname": host,
                    "host": address,
                    "port": port,
                    "family": family,
                    "proto": 0,
                    "flags": 0,
                }
                for address in self.addresses
            ]

        async def close(self):
            return None

    async def run() -> None:
        fake_ip = _PublicOnlyResolver(_Resolver(["fc00::33", "198.18.0.52"]))
        resolved = await fake_ip.resolve("huggingface.co", 443, 0)
        assert [item["host"] for item in resolved] == ["198.18.0.52"]

        mixed = _PublicOnlyResolver(_Resolver(["10.0.0.8", "8.8.8.8"]))
        resolved = await mixed.resolve("huggingface.co", 443, 0)
        assert [item["host"] for item in resolved] == ["8.8.8.8"]

        private = _PublicOnlyResolver(_Resolver(["127.0.0.1", "192.168.1.8"]))
        with pytest.raises(OSError, match="私有网络地址"):
            await private.resolve("huggingface.co", 443, 0)

    asyncio.run(run())


def test_download_success_is_atomic_and_reusable(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        service, asset, payload, session = _service(tmp_path, monkeypatch)
        started = await service.start_download(asset.id)
        download_id = started["download"]["id"]
        result = await _wait_terminal(service, download_id)
        assert result["state"] == "completed"
        assert result["bytes_downloaded"] == len(payload)
        target = asset_file_path(asset, asset.files[0])
        assert target.read_bytes() == payload
        assert not list(target.parent.glob("*.part"))
        assert not list(target.parent.glob(".download-*"))
        assert session.closed is True

        reused = await service.start_download(asset.id)
        assert reused["download"]["state"] == "completed"
        await service.shutdown()

    asyncio.run(run())


def test_download_requests_are_deduplicated_and_cancelled_cleanly(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
        asset, payload = _asset()
        session = _GateSession(payload)
        service = ModelDownloadService(
            assets=[asset],
            session_factory=lambda: session,
            url_factory=lambda _asset, _file: "https://huggingface.co/fixture/file",
        )
        first = await service.start_download(asset.id)
        download_id = first["download"]["id"]
        await asyncio.wait_for(session.content.started.wait(), timeout=1)
        duplicate = await service.start_download(asset.id)
        assert duplicate["deduplicated"] is True
        assert duplicate["download"]["id"] == download_id

        canceled = await service.cancel_download(download_id)
        assert canceled["download"]["state"] == "canceled"
        target = asset_file_path(asset, asset.files[0])
        assert not target.exists()
        assert not list(target.parent.glob("*.part"))
        assert not list(target.parent.glob(".download-*"))
        await service.shutdown()

    asyncio.run(run())


def test_download_checksum_failure_leaves_no_final_file(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        # The manifest expects ``good`` while the transport serves ``bad``.
        monkeypatch.setenv("ANIMA_CAPTIONER_MODELS_ROOT", str(tmp_path / "models"))
        asset, _payload = _asset(b"good")
        session = _Session(b"bad!")
        service = ModelDownloadService(
            assets=[asset],
            session_factory=lambda: session,
            url_factory=lambda _asset, _file: "https://huggingface.co/fixture/file",
        )
        started = await service.start_download(asset.id)
        result = await _wait_terminal(service, started["download"]["id"])
        assert result["state"] == "error"
        assert "完整性校验" in result["error"]
        target = asset_file_path(asset, asset.files[0])
        assert not target.exists()
        assert not list(target.parent.glob("*.part"))
        assert not list(target.parent.glob(".download-*"))
        await service.shutdown()

    asyncio.run(run())


class _RouteService:
    def __init__(self, service: ModelDownloadService):
        self.service = service

    async def list_model_assets(self):
        return await self.service.list_assets()

    async def get_model_asset(self, asset_id):
        return await self.service.get_asset(asset_id)

    async def start_model_download(self, asset_id):
        return await self.service.start_download(asset_id)

    def list_model_downloads(self):
        return self.service.list_downloads()

    def get_model_download(self, download_id):
        return self.service.get_download(download_id)

    async def cancel_model_download(self, download_id):
        return await self.service.cancel_download(download_id)


def test_model_asset_routes_support_both_prefixes(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        service, asset, _payload, _session = _service(tmp_path, monkeypatch)
        app = web.Application()
        app["tagging_service"] = _RouteService(service)
        setup_tagging_routes(app)
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                for prefix in ("/api/captioning", "/api/tagging"):
                    listed = await client.get(f"{prefix}/model-assets")
                    assert listed.status == 200
                    body = await listed.json()
                    assert body["manifest_version"] == 1
                    assert body["assets"][0]["id"] == asset.id
                    assert "directory" in body["assets"][0]
                    assert "/tmp/" not in str(body)

                started = await client.post(f"/api/captioning/model-assets/{asset.id}/download")
                assert started.status == 202
                download_id = (await started.json())["download"]["id"]
                for _ in range(100):
                    polled = await client.get(f"/api/tagging/downloads/{download_id}")
                    payload = await polled.json()
                    if payload["download"]["state"] == "completed":
                        break
                    await asyncio.sleep(0.005)
                else:
                    raise AssertionError("route download did not complete")
                missing = await client.get("/api/captioning/downloads/not-found")
                assert missing.status == 404
        await service.shutdown()

    asyncio.run(run())
