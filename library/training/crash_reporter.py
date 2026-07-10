"""Fatal exception reporting into progress.jsonl for daemon-friendly diagnostics."""

from __future__ import annotations

import os
import sys


def _install_crash_reporter(argv: list[str]) -> None:
    """Record a fatal startup/training exception into ``--progress_jsonl``.

    The daemon launches us windowless under ``pythonw.exe``; that interpreter
    drops the child's stdout/stderr (only the ``accelerate launch`` *parent*'s
    output reaches ``stdout.log``), so an uncaught traceback here is lost and the
    daemon falls back to a generic "process exited (code=1)" with nothing
    actionable. ``progress.jsonl`` is written by path, not via the dead std
    streams, so it survives — and it's what the daemon already reads to diagnose
    a job (``manager._finalize_from_exit`` → ``run_end.error``).

    ``run_scope`` already emits ``run_end(error=…)`` for failures inside the
    training loop, but only *after* ``ProgressSink.run_start`` has fired — late
    in ``train()``. Errors before that (latent/TE cache incomplete, config or
    dataset build, model load) escape it entirely. This excepthook is the
    catch-all: it appends a ``run_end`` error event for any uncaught exception,
    wherever it's raised, so the GUI's finish banner shows the real cause.
    """
    path = None
    for i, tok in enumerate(argv):
        if tok == "--progress_jsonl" and i + 1 < len(argv):
            path = argv[i + 1]
        elif tok.startswith("--progress_jsonl="):
            path = tok.split("=", 1)[1]
    if not path or path.strip().lower() in ("", "none", "off"):
        return

    import json as _json

    prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        # KeyboardInterrupt is a clean stop, handled by run_scope/the daemon's
        # stop_requested path — don't mislabel it an error.
        if not issubclass(exc_type, KeyboardInterrupt):
            try:
                # Dedupe: run_scope may already have written the terminal event
                # for an in-loop failure; don't append a second one.
                already_ended = False
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                last = line
                    try:
                        already_ended = _json.loads(last).get("ev") == "run_end"
                    except (NameError, ValueError):
                        already_ended = False
                if not already_ended:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(
                            _json.dumps(
                                {
                                    "ev": "run_end",
                                    "status": "error",
                                    "final_step": -1,
                                    "error": f"{exc_type.__name__}: {exc}",
                                }
                            )
                            + "\n"
                        )
            except Exception:  # noqa: BLE001 — reporting must never mask the crash
                pass
        prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook

