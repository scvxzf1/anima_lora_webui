from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import toml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import library.env as library_env
from web.routes import config as config_routes
from web.routes import setup_config_routes, setup_settings_routes
from web.services import settings_service
from tests.web_config_test_support import (
    _patch_config_service_paths,
    _write_minimal_config_tree,
)


def _model(item_name: str) -> dict[str, str]:
    return {
        "id": "isolated-model",
        "name": item_name,
        "model_family": "krea2_raw",
        "pretrained_model_name_or_path": "models/krea2/dit.safetensors",
        "qwen3": "models/krea2/qwen3vl.safetensors",
        "vae": "models/krea2/vae.safetensors",
    }


def test_s2_config_dataset_and_prompts_roundtrip_over_isolated_http_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_path = configs / "imported" / "lora.toml"
    initial_train = (
        'output_name = "before"\n'
        "max_train_steps = 12\n"
        'unknown_s2_key = "preserve-me"\n'
    )
    train_path.write_text(initial_train, encoding="utf-8")

    async def run() -> None:
        app = web.Application()
        setup_config_routes(app)
        async with TestClient(TestServer(app)) as client:
            preview_response = await client.post(
                "/api/config/raw/patch-preview",
                json={
                    "file": "configs/imported/lora.toml",
                    "values": {"output_name": "after", "max_train_steps": 24},
                },
            )
            assert preview_response.status == 200
            preview = await preview_response.json()
            assert set(preview["changed"]) == {"output_name", "max_train_steps"}
            assert train_path.read_text(encoding="utf-8") == initial_train

            save_response = await client.patch(
                "/api/config/raw",
                json={
                    "file": "configs/imported/lora.toml",
                    "values": {"output_name": "after", "max_train_steps": 24},
                },
            )
            assert save_response.status == 200
            saved = await save_response.json()
            assert saved["content"] == train_path.read_text(encoding="utf-8")
            assert 'unknown_s2_key = "preserve-me"' in saved["content"]

            reloaded = await client.get(
                "/api/config/raw", params={"file": "configs/imported/lora.toml"}
            )
            assert reloaded.status == 200
            assert (await reloaded.json())["content"] == saved["content"]

            save_as = await client.post(
                "/api/config/raw/save-as",
                json={"file": "configs/imported/copy.toml", "content": saved["content"]},
            )
            assert save_as.status == 200
            collision = await client.post(
                "/api/config/raw/save-as",
                json={
                    "file": "configs/imported/copy.toml",
                    "content": "replacement = true\n",
                },
            )
            assert collision.status == 400
            assert 'output_name = "after"' in (
                configs / "imported" / "copy.toml"
            ).read_text(encoding="utf-8")

            rename = await client.post(
                "/api/config/raw/rename",
                json={
                    "source": "configs/imported/copy.toml",
                    "target": "configs/imported/renamed.toml",
                },
            )
            assert rename.status == 200
            assert not (configs / "imported" / "copy.toml").exists()
            assert (configs / "imported" / "renamed.toml").exists()

            prompts = "# character A\n\n  masterpiece, best quality  \nsolo\n"
            prompt_response = await client.put(
                "/api/config/sample-prompts",
                json={
                    "file": "configs/sample-prompts/imported/lora.txt",
                    "content": prompts,
                },
            )
            assert prompt_response.status == 200
            prompt_reload = await client.get(
                "/api/config/sample-prompts",
                params={"file": "configs/sample-prompts/imported/lora.txt"},
            )
            assert prompt_reload.status == 200
            assert (await prompt_reload.json())["content"] == prompts
            assert (configs / "sample-prompts/imported/lora.txt").read_text(
                encoding="utf-8"
            ) == prompts

            escaped_prompt = await client.put(
                "/api/config/sample-prompts",
                json={"file": "configs/../outside.txt", "content": "nope\n"},
            )
            assert escaped_prompt.status == 400
            assert not (tmp_path / "outside.txt").exists()

            dataset_file = "configs/datasets/isolated.toml"
            dataset_payload = {
                "file": dataset_file,
                "overwrite": False,
                "defaults": {
                    "resolution": 1024,
                    "batch_size": 1,
                    "enable_bucket": True,
                    "prior_loss_weight": 1.25,
                },
                "datasets": [
                    {
                        "source_dir": "image_dataset/first",
                        "image_dir": "post/first",
                        "cache_dir": "cache/first",
                        "num_repeats": 5,
                        "is_reg": False,
                        "settings": {"caption_extension": ".txt"},
                    },
                    {
                        "source_dir": "image_dataset/second",
                        "image_dir": "post/second",
                        "cache_dir": "cache/second",
                        "num_repeats": 1,
                        "is_reg": False,
                        "settings": {"caption_extension": ".txt"},
                    },
                ],
                "stage_schedule_enabled": True,
                "stage_schedule": [
                    {"name": "all", "subset_index": 0, "start_pct": 0, "end_pct": 1}
                ],
            }
            dataset_response = await client.put(
                "/api/config/dataset-presets", json=dataset_payload
            )
            assert dataset_response.status == 200, await dataset_response.text()

            dataset_tab_a = await client.get(
                "/api/config/dataset-presets/read", params={"file": dataset_file}
            )
            dataset_tab_b = await client.get(
                "/api/config/dataset-presets/read", params={"file": dataset_file}
            )
            revision_a = (await dataset_tab_a.json())["revision"]
            revision_b = (await dataset_tab_b.json())["revision"]
            assert revision_a == revision_b
            first_dataset_write = await client.put(
                "/api/config/dataset-presets",
                json={**dataset_payload, "overwrite": True, "revision": revision_a,
                      "defaults": {**dataset_payload["defaults"], "prior_loss_weight": 1.5}},
            )
            assert first_dataset_write.status == 200
            stale_dataset_write = await client.put(
                "/api/config/dataset-presets",
                json={**dataset_payload, "overwrite": True, "revision": revision_b,
                      "defaults": {**dataset_payload["defaults"], "prior_loss_weight": 2.0}},
            )
            assert stale_dataset_write.status == 409
            assert "prior_loss_weight = 1.5" in (
                configs / "datasets" / "isolated.toml"
            ).read_text(encoding="utf-8")

            group_response = await client.post(
                "/api/config/file-groups",
                json={"label": "S2 isolated", "kind": "dataset"},
            )
            assert group_response.status == 200
            group_id = (await group_response.json())["group"]["id"]
            place_response = await client.post(
                "/api/config/file-groups/place",
                json={
                    "target": "file",
                    "file": dataset_file,
                    "group": group_id,
                    "order": [dataset_file],
                },
            )
            assert place_response.status == 200

            before_failed_apply = train_path.read_text(encoding="utf-8")
            from web.services.config import dataset_presets_api

            with monkeypatch.context() as patcher:
                patcher.setattr(
                    dataset_presets_api,
                    "save_raw_file",
                    lambda *_args, **_kwargs: (False, "fixture write failure", []),
                )
                failed_apply = await client.post(
                    "/api/config/dataset-presets/apply",
                    json={
                        "dataset_file": dataset_file,
                        "train_file": "configs/imported/lora.toml",
                    },
                )
            assert failed_apply.status == 400
            assert train_path.read_text(encoding="utf-8") == before_failed_apply

            apply_response = await client.post(
                "/api/config/dataset-presets/apply",
                json={
                    "dataset_file": dataset_file,
                    "train_file": "configs/imported/lora.toml",
                },
            )
            assert apply_response.status == 200
            applied = await apply_response.json()
            assert applied["values"]["dataset_config"] == dataset_file
            assert 'source_image_dir = "image_dataset/first"' in train_path.read_text(
                encoding="utf-8"
            )

            preset_reload = await client.get(
                "/api/config/dataset-presets/read", params={"file": dataset_file}
            )
            assert preset_reload.status == 200
            reloaded_dataset = await preset_reload.json()
            assert [row["source_dir"] for row in reloaded_dataset["datasets"]] == [
                "image_dataset/first",
                "image_dataset/second",
            ]
            group_text = (configs / "web-file-groups.toml").read_text(encoding="utf-8")
            assert dataset_file in group_text

    asyncio.run(run())


def test_s2_model_revision_conflict_and_configs_root_switch_do_not_move_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    (tmp_path / ".anima-webui-settings.toml").write_text(
        '[paths]\nconfigs_root = "configs"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(library_env, "project_root", lambda: tmp_path)
    monkeypatch.delenv("ANIMA_CONFIGS_ROOT", raising=False)
    settings_file = configs / "web-ui-settings.toml"
    settings_file.write_text(
        '[global]\noutput_root = "output/runs"\nunknown_setting = "keep"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    train_path = configs / "imported" / "lora.toml"
    train_path.write_text('output_name = "stays-here"\n', encoding="utf-8")
    dataset_path = configs / "datasets" / "stays-here.toml"
    dataset_path.write_text('name = "stays-here"\n', encoding="utf-8")
    old_train = train_path.read_text(encoding="utf-8")
    old_dataset = dataset_path.read_text(encoding="utf-8")

    async def run() -> None:
        app = web.Application()
        setup_config_routes(app)
        setup_settings_routes(app)
        async with TestClient(TestServer(app)) as client:
            initial = await client.get("/api/settings/model-configs")
            assert initial.status == 200
            model_state = await initial.json()
            first_body = {
                "revision": model_state["revision"],
                "default_id": "isolated-model",
                "items": [_model("First write")],
                "groups": [
                    {
                        "id": "isolated-group",
                        "label": "Isolated",
                        "item_ids": ["isolated-model"],
                    }
                ],
            }
            first_write = await client.put("/api/settings/model-configs", json=first_body)
            assert first_write.status == 200
            old_settings = settings_file.read_text(encoding="utf-8")
            stale_body = {**first_body, "items": [_model("Stale write")]}
            conflict = await client.put("/api/settings/model-configs", json=stale_body)
            assert conflict.status == 409
            current = await client.get("/api/settings/model-configs")
            assert (await current.json())["items"][0]["name"] == "First write"

            settings_tab_a = await client.get("/api/settings/global")
            settings_tab_b = await client.get("/api/settings/global")
            settings_revision = (await settings_tab_a.json())["revision"]
            assert settings_revision == (await settings_tab_b.json())["revision"]
            first_settings_write = await client.put(
                "/api/settings/global",
                json={"revision": settings_revision, "ui_scale": 125},
            )
            stale_settings_write = await client.put(
                "/api/settings/global",
                json={"revision": settings_revision, "ui_scale": 150},
            )
            assert first_settings_write.status == 200
            assert stale_settings_write.status == 409
            assert toml.loads(settings_file.read_text(encoding="utf-8"))["global"]["ui_scale"] == 125
            old_settings = settings_file.read_text(encoding="utf-8")

            switched = await client.put(
                "/api/settings/global", json={"configs_root": "external-configs"}
            )
            assert switched.status == 200
            switch_payload = await switched.json()
            assert switch_payload["requires_reload"] is True
            assert switch_payload["configs_root"] == "external-configs"

            external = tmp_path / "external-configs"
            assert (external / "web-ui-settings.toml").exists()
            assert toml.loads(
                (external / "web-ui-settings.toml").read_text(encoding="utf-8")
            )["model_config_library"]["items"][0]["name"] == "First write"
            assert settings_file.read_text(encoding="utf-8") == old_settings
            assert train_path.read_text(encoding="utf-8") == old_train
            assert dataset_path.read_text(encoding="utf-8") == old_dataset
            assert not (external / "imported" / "lora.toml").exists()
            assert not (external / "datasets" / "stays-here.toml").exists()

            dataset_list = await client.get("/api/config/dataset-presets")
            assert dataset_list.status == 200
            assert (await dataset_list.json())["presets"] == []
            switched_models = await client.get("/api/settings/model-configs")
            assert (await switched_models.json())["items"][0]["name"] == "First write"

    asyncio.run(run())


def test_s2_http_server_failure_does_not_change_fixture_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    path = configs / "imported" / "lora.toml"
    original = 'output_name = "stable"\n'
    path.write_text(original, encoding="utf-8")
    calls = 0

    def fail_before_write(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError("fixture disk failure")

    monkeypatch.setattr(config_routes, "patch_raw_file_values", fail_before_write)

    async def run() -> None:
        app = web.Application()
        setup_config_routes(app)
        async with TestClient(TestServer(app)) as client:
            response = await client.patch(
                "/api/config/raw",
                json={
                    "file": "configs/imported/lora.toml",
                    "values": {"output_name": "should-not-persist"},
                },
            )
            assert response.status == 500

    asyncio.run(run())
    assert calls == 1
    assert path.read_text(encoding="utf-8") == original


def test_s2_raw_and_prompt_writes_reject_stale_two_tab_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    path = configs / "imported" / "lora.toml"
    path.write_text('output_name = "baseline"\n', encoding="utf-8")
    prompt_path = configs / "sample_prompts.txt"
    prompt_path.write_text("baseline prompt\n", encoding="utf-8")

    async def run() -> None:
        app = web.Application()
        setup_config_routes(app)
        async with TestClient(TestServer(app)) as client:
            first_tab = await client.get(
                "/api/config/raw", params={"file": "configs/imported/lora.toml"}
            )
            second_tab = await client.get(
                "/api/config/raw", params={"file": "configs/imported/lora.toml"}
            )
            first_snapshot = await first_tab.json()
            second_snapshot = await second_tab.json()
            assert first_snapshot["content"] == second_snapshot["content"]
            assert first_snapshot["revision"] == second_snapshot["revision"]

            first_write = await client.patch(
                "/api/config/raw",
                json={
                    "file": "configs/imported/lora.toml",
                    "values": {"output_name": "first-tab"},
                    "revision": first_snapshot["revision"],
                },
            )
            second_write = await client.patch(
                "/api/config/raw",
                json={
                    "file": "configs/imported/lora.toml",
                    "values": {"output_name": "second-tab"},
                    "revision": second_snapshot["revision"],
                },
            )
            assert first_write.status == 200
            assert second_write.status == 409
            stale_raw_put = await client.put(
                "/api/config/raw",
                json={"file": "configs/imported/lora.toml", "content": 'output_name = "stale-raw"\n', "revision": second_snapshot["revision"]},
            )
            assert stale_raw_put.status == 409
            assert 'output_name = "first-tab"' in path.read_text(encoding="utf-8")
            refreshed = await client.get(
                "/api/config/raw", params={"file": "configs/imported/lora.toml"}
            )
            refreshed_snapshot = await refreshed.json()
            assert 'output_name = "first-tab"' in refreshed_snapshot["content"]
            assert refreshed_snapshot["revision"] == (await first_write.json())["revision"]

            first_prompts = await client.get(
                "/api/config/sample-prompts",
                params={"file": "configs/sample_prompts.txt"},
            )
            second_prompts = await client.get(
                "/api/config/sample-prompts",
                params={"file": "configs/sample_prompts.txt"},
            )
            first_prompt_snapshot = await first_prompts.json()
            second_prompt_snapshot = await second_prompts.json()
            assert first_prompt_snapshot["revision"] == second_prompt_snapshot["revision"]
            first_prompt_write = await client.put(
                "/api/config/sample-prompts",
                json={"file": "configs/sample_prompts.txt", "content": "# keep\nfirst prompt\n", "revision": first_prompt_snapshot["revision"]},
            )
            stale_prompt_write = await client.put(
                "/api/config/sample-prompts",
                json={"file": "configs/sample_prompts.txt", "content": "second prompt\n", "revision": second_prompt_snapshot["revision"]},
            )
            assert first_prompt_write.status == 200
            assert stale_prompt_write.status == 409
            assert prompt_path.read_text(encoding="utf-8") == "# keep\nfirst prompt\n"
            prompt_refresh = await client.get("/api/config/sample-prompts", params={"file": "configs/sample_prompts.txt"})
            assert (await prompt_refresh.json())["revision"] == (await first_prompt_write.json())["revision"]

            missing_fork = await client.get(
                "/api/config/sample-prompts",
                params={"file": "configs/sample-prompts/imported/lora.txt"},
            )
            missing_revision = (await missing_fork.json())["revision"]
            created_fork = await client.put(
                "/api/config/sample-prompts",
                json={"train_config_file": "configs/imported/lora.toml", "content": "first fork\n", "revision": missing_revision},
            )
            stale_fork = await client.put(
                "/api/config/sample-prompts",
                json={"train_config_file": "configs/imported/lora.toml", "content": "second fork\n", "revision": missing_revision},
            )
            assert created_fork.status == 200
            assert stale_fork.status == 409
            assert (configs / "sample-prompts/imported/lora.txt").read_text(encoding="utf-8") == "first fork\n"

    asyncio.run(run())
