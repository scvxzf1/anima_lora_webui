"""Route registration."""

from aiohttp import web

from web.routes.config import setup_config_routes
from web.routes.preview import setup_preview_routes
from web.routes.settings import setup_settings_routes
from web.routes.training import setup_training_routes


def setup_routes(app: web.Application) -> None:
    setup_settings_routes(app)
    setup_config_routes(app)
    setup_preview_routes(app)
    setup_training_routes(app)
