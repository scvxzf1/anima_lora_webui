"""Isolated real-HTTP mask fixture. No training service or user-data access.

Run: .venv/bin/python -m tests.manual_mask_hot_server --port 20541
"""

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from PIL import Image, ImageDraw

from tests.web_config_test_support import _patch_config_service_paths, _write_minimal_config_tree
from web.routes.config import setup_config_routes
from web.routes.mask_editor import setup_mask_editor_routes
from web.routes.tagging import setup_tagging_routes
from web.server import next_index_handler, static_handler
from web.services import config_service


def create_fixture(root: Path):
    _write_minimal_config_tree(root)
    for name in ["source", "training", "source-2", "training-2"]:
        (root / name).mkdir()
    for index, color in enumerate(["#bd4b67", "#3676a1", "#6e8950"]):
        image = Image.new("RGB", (768, 768), "#dfe5e6")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 568, 768, 768), fill="#bac9cb")
        draw.polygon([(242, 146), (310, 125), (458, 125), (526, 146), (650, 260),
                      (554, 348), (506, 310), (526, 650), (242, 650), (262, 310),
                      (214, 348), (118, 260)], fill=color)
        draw.arc((310, 85, 458, 195), 0, 180, fill="#f0f0ef", width=18)
        draw.line((265, 625, 501, 625), fill="#f0f0ef", width=5)
        for name in ["source", "training", "source-2", "training-2"]:
            image.save(root / name / f"garment-{index + 1:02}.png")
    config_service.save_dataset_preset("configs/datasets/mask-hot-test.toml", [{
        "source_dir": "source", "image_dir": "training", "cache_dir": "cache",
        "mask_mode": "none", "num_repeats": 1,
    }, {
        "source_dir": "source-2", "image_dir": "training-2", "cache_dir": "cache-2",
        "mask_mode": "none", "num_repeats": 1,
    }], {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=20541)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="anima-mask-hot-"))
    patch = pytest.MonkeyPatch()
    _patch_config_service_paths(patch, root)
    create_fixture(root)
    app = web.Application(client_max_size=32 * 1024 * 1024)
    setup_config_routes(app)
    setup_mask_editor_routes(app)
    setup_tagging_routes(app)
    app["tagging_service"] = SimpleNamespace(
        list_profiles=lambda: {"ok": True, "active_profile_id": "", "profiles": [], "provider_types": []},
        list_jobs=lambda: {"ok": True, "jobs": []},
        list_prompt_presets=lambda: {"presets": []},
    )

    async def settings(_request):
        return web.json_response({"ui_scale": 100})

    async def fixture(_request):
        return web.json_response({"root": str(root), "isolated": True})

    app.router.add_get("/api/settings/global", settings)
    app.router.add_get("/api/mask-hot-fixture", fixture)
    app.router.add_get("/next/{path:.*}", next_index_handler)
    app.router.add_get("/static/{path:.*}", static_handler)
    print(f"Isolated mask fixture: {root}", flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
