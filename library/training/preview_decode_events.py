"""Best-effort preview-only event persistence; never a training OOM signal."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def emit_decode_event(sink, event, **fields):
    if sink is None:
        return
    try:
        sink({"event": event, **fields})
    except Exception:
        logger.warning("Unable to write training preview VAE decode event", exc_info=True)


def decode_event_sink(output_dir, latent_path):
    destination = Path(output_dir) / "sample" / "vae-decode-events.jsonl"

    def append(event):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "latent_filename": Path(latent_path).name, **event}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    return append
