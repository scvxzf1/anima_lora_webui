from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from web.routes.tagging import setup_tagging_routes
from web.services.tagging import TaggingService, profiles, settings


def _patch_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "configs" / "captioning" / "settings.toml")
    monkeypatch.setattr(settings, "SECRETS_FILE", tmp_path / ".anima-captioning-secrets.toml")
    for name in (*settings._ENV_BASE_URL, *settings._ENV_MODEL, *settings._ENV_KEY):
        monkeypatch.delenv(name, raising=False)


def test_profiles_keep_legacy_settings_and_mask_profile_keys(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    settings.save_settings({"base_url": "https://legacy.example/v1", "model": "legacy"})

    initial = profiles.list_profiles()
    assert initial["active_profile_id"] == profiles.LEGACY_PROFILE_ID
    assert initial["profiles"][0]["provider"] == "openai_compatible"

    created = profiles.create_profile(
        {
            "id": "openai-secondary",
            "name": "备用视觉 API",
            "provider": "openai_compatible",
            "config": {"base_url": "https://vision.example/v1", "model": "vision-2"},
            "api_key": "profile-secret",
        }
    )
    assert created["active_profile_id"] == profiles.LEGACY_PROFILE_ID
    assert all("api_key" not in profile for profile in created["profiles"])
    assert profiles.get_profile_api_key("openai-secondary") == "profile-secret"

    activated = profiles.activate_profile("openai-secondary")
    assert activated["active_profile_id"] == "openai-secondary"
    public = settings.get_public_settings()
    assert public["profile_id"] == "openai-secondary"
    assert public["model"] == "vision-2"
    assert "profile-secret" not in str(public)

    settings.save_settings({"model": "vision-3", "api_key": ""})
    assert profiles.get_effective_settings("openai-secondary")["model"] == "vision-3"
    assert profiles.get_profile_api_key("openai-secondary") == "profile-secret"


def test_local_profiles_are_reserved_without_claiming_to_be_available(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    created = profiles.create_profile(
        {
            "id": "wd14-local",
            "name": "WD14 本地预留",
            "provider": "wd14",
            "config": {"asset_id": "wd14-eva02-large-v3", "device": "cpu", "batch_size": 4},
        }
    )
    profile = next(item for item in created["profiles"] if item["id"] == "wd14-local")
    assert profile["kind"] == "local"
    assert profile["status"] == "not_installed"
    assert profile["available"] is False
    assert not (tmp_path / "models").exists()


def test_local_profile_normalizes_target_project_model_ids(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    created = profiles.create_profile(
        {
            "id": "vit-local",
            "name": "WD14 ViT",
            "provider": "wd14",
            "config": {"model_id": "SmilingWolf/wd-vit-tagger-v3"},
        }
    )
    profile = next(item for item in created["profiles"] if item["id"] == "vit-local")
    assert profile["asset_id"] == "wd14-vit-v3"
    assert profile["config"]["asset_id"] == "wd14-vit-v3"


def test_local_profile_persists_one_explicit_gpu_index(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    created = profiles.create_profile(
        {
            "id": "wd14-gpu",
            "name": "WD14 GPU 2",
            "provider": "wd14",
            "config": {
                "asset_id": "wd14-eva02-large-v3",
                "device": "cuda",
                "gpu_index": "2",
            },
        }
    )
    profile = next(item for item in created["profiles"] if item["id"] == "wd14-gpu")
    assert profile["config"]["gpu_index"] == 2
    assert profiles.get_effective_settings("wd14-gpu")["gpu_index"] == 2

    updated = profiles.update_profile("wd14-gpu", {"config": {"device": "cpu"}})
    profile = next(item for item in updated["profiles"] if item["id"] == "wd14-gpu")
    assert "gpu_index" not in profile["config"]


@pytest.mark.parametrize("value", [-1, True, "1.5", "gpu0"])
def test_local_profile_rejects_invalid_gpu_index(tmp_path, monkeypatch, value) -> None:
    _patch_paths(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="GPU 序号"):
        profiles.create_profile(
            {
                "id": "wd14-invalid-gpu",
                "name": "WD14 invalid GPU",
                "provider": "wd14",
                "config": {
                    "asset_id": "wd14-eva02-large-v3",
                    "device": "cuda",
                    "gpu_index": value,
                },
            }
        )


def test_unavailable_local_profile_cannot_be_activated(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    created = profiles.create_profile(
        {
            "id": "wd14-local",
            "name": "WD14 本地",
            "provider": "wd14",
            "config": {"asset_id": "wd14-eva02-large-v3"},
        }
    )
    assert next(item for item in created["profiles"] if item["id"] == "wd14-local")["available"] is False
    with pytest.raises(ValueError, match="当前不可用"):
        profiles.activate_profile("wd14-local")


def test_local_profile_activation_rechecks_full_asset_hash(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)
    profiles.create_profile(
        {
            "id": "wd14-local",
            "name": "WD14 本地",
            "provider": "wd14",
            "config": {"asset_id": "fixture-asset"},
        }
    )

    from web.services.tagging import model_assets

    monkeypatch.setattr(
        profiles,
        "_public_profile",
        lambda _profile: {"available": True, "kind": "local"},
    )

    class FixtureAsset:
        provider = "wd14"

    checks = []
    monkeypatch.setattr(model_assets, "get_model_asset", lambda _asset_id: FixtureAsset())

    def inspect(_asset, *, verify_hash=True):
        checks.append(verify_hash)
        return {"state": "corrupt"}

    monkeypatch.setattr(model_assets, "inspect_asset", inspect)
    saves = []
    monkeypatch.setattr(profiles, "_save_profiles", lambda *_args: saves.append(True))

    with pytest.raises(ValueError, match="模型文件校验失败"):
        profiles.activate_profile("wd14-local")

    assert checks == [True]
    assert saves == []


def test_profile_routes_are_available_under_both_prefixes(tmp_path, monkeypatch) -> None:
    _patch_paths(tmp_path, monkeypatch)

    async def run() -> None:
        app = web.Application()
        app["tagging_service"] = TaggingService(app)
        setup_tagging_routes(app)
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                for prefix in ("/api/captioning", "/api/tagging"):
                    listed = await client.get(f"{prefix}/profiles")
                    assert listed.status == 200
                    payload = await listed.json()
                    assert payload["ok"] is True
                    assert payload["profiles"]
                    assert "provider_types" in payload

                created = await client.post(
                    "/api/captioning/profiles",
                    json={
                        "id": "route-api",
                        "name": "路由 API",
                        "provider": "openai_compatible",
                        "config": {"base_url": "https://route.example/v1", "model": "route-model"},
                        "api_key": "route-secret",
                    },
                )
                assert created.status == 201
                body = await created.json()
                assert body["profile"]["id"] == "route-api"
                assert "route-secret" not in str(body)

                activated = await client.post("/api/tagging/profiles/route-api/activate", json={})
                assert activated.status == 200
                assert (await activated.json())["active_profile_id"] == "route-api"

                removed = await client.delete("/api/captioning/profiles/route-api")
                assert removed.status == 200

    asyncio.run(run())
