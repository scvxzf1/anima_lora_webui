"""tqdm progress-bar parsing for QProcess output streams.

Both ConfigTab and PreprocessingTab pipe a child process's stdout/stderr
through a small QProgressBar at the top of the tab. The parsing logic
(matching tqdm's textual format and computing s/step from the first
completed step) is shared here so the two tabs don't drift.

Use as::

    self.tracker = TqdmProgressTracker(self.progress)
    ...
    line = parts[0]
    if not self.tracker.feed(line):
        self._log(line + "\\n")
"""

from __future__ import annotations

import json
import os
import re
import time

from PySide6.QtWidgets import QProgressBar

# Matches tqdm lines like:
#   "Denoising steps:  40%|####      | 12/30 [00:12<00:34,  2.50it/s]"
# The trailing "[...]" block carries the rate as either "X.XXit/s" or
# "X.XXs/it"; both are captured optionally so non-timed bars still parse.
TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?:[^\[]*\[[^\]]*?(?P<rate>[\d.]+)(?P<unit>it/s|s/it)[^\]]*\])?"
)


def make_progress_bar() -> QProgressBar:
    """Build a QProgressBar styled to match the rest of the GUI.

    Returns a hidden bar — the tracker shows it on the first parsed update
    and ``TqdmProgressTracker.reset`` hides it again at run-end.
    """
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setFormat("")
    bar.setVisible(False)
    bar.setStyleSheet(
        "QProgressBar { border: 1px solid #444; border-radius: 3px;"
        " text-align: center; padding: 1px; font-size: 11px; }"
        "QProgressBar::chunk { background: #27ae60; }"
    )
    return bar


class TqdmProgressTracker:
    """Parses tqdm output lines and drives a QProgressBar.

    Holds an anchor (timestamp + step) seeded from the *first completed*
    step of each new bar, so reported s/step doesn't include warm-up
    overhead (model load, compile, dataset scan).
    """

    def __init__(self, bar: QProgressBar) -> None:
        self._bar = bar
        # (monotonic_anchor_time, anchor_step, label, total)
        self._anchor: tuple[float, int, str, int] | None = None

    def reset(self) -> None:
        """Zero the bar, hide it, drop the rate anchor."""
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._bar.setVisible(False)
        self._anchor = None

    def mark_starting(self, label: str) -> None:
        """Show the bar in indeterminate "busy" mode (Qt animates range 0-0).

        Used between subprocess launch and the first tqdm line so Windows
        doesn't flag the GUI as "Not Responding" during the multi-second
        torch/accelerate import inside the child.
        """
        self._bar.setRange(0, 0)  # indeterminate / marquee
        self._bar.setFormat(label)
        self._bar.setVisible(True)
        self._anchor = None

    def feed(self, line: str) -> bool:
        """Try to parse *line* as a tqdm update. Returns True if matched.

        The caller passes non-matching lines to its log widget instead.
        """
        m = TQDM_RE.search(line)
        if not m:
            return False
        cur = int(m.group("cur"))
        tot = int(m.group("tot"))
        label = m.group("label").strip() or "progress"
        rate_str = self._update_rate(label, cur, tot)
        if tot > 0:
            # Leaving indeterminate mode: setRange(0, tot) clears the marquee
            # animation; first determinate update replaces the "Starting…"
            # label seamlessly.
            self._bar.setRange(0, tot)
            self._bar.setValue(cur)
            self._bar.setFormat(f"{label}: {cur}/{tot} (%p%){rate_str}")
            if not self._bar.isVisible():
                self._bar.setVisible(True)
        return True

    def _update_rate(self, label: str, cur: int, tot: int) -> str:
        now = time.monotonic()
        anchor = self._anchor
        # New bar (label/total changed, or progress rewound) → drop anchor.
        if (
            anchor is None
            or anchor[2] != label
            or anchor[3] != tot
            or cur < anchor[1]
        ):
            if cur >= 1:
                self._anchor = (now, cur, label, tot)
            else:
                self._anchor = None
            return ""
        anchor_time, anchor_step, _, _ = anchor
        steps = cur - anchor_step
        if steps <= 0:
            return ""
        spi = (now - anchor_time) / steps
        remaining = tot - cur
        if remaining <= 0:
            return f" — {spi:.2f}s/step"
        return f" — {spi:.2f}s/step — ETA {_format_duration(remaining * spi)}"


class JsonlProgressReader:
    """Drive a QProgressBar from a training progress.jsonl stream."""

    def __init__(self, bar: QProgressBar) -> None:
        self._bar = bar
        self._path: str | None = None
        self._pos = 0
        self._total_steps = 0
        self._active = False
        self._anchor: tuple[float, int] | None = None

    @property
    def active(self) -> bool:
        return self._active

    def watch(self, path: str | None) -> None:
        self._path = path
        self._pos = 0
        self._total_steps = 0
        self._active = False
        self._anchor = None

    def reset(self) -> None:
        self.watch(None)

    def poll(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                self._pos -= len(line.encode("utf-8"))
                return
            if isinstance(event, dict):
                self._consume(event)

    def _consume(self, event: dict) -> None:
        kind = event.get("ev")
        if kind == "run_start":
            self._active = True
            self._total_steps = int(event.get("total_steps") or 0)
            self._anchor = None
            self._bar.setRange(0, self._total_steps or 0)
            self._bar.setValue(0)
            self._bar.setFormat("starting...")
            self._bar.setVisible(True)
            return
        if kind == "step":
            self._active = True
            self._update_bar(int(event.get("global_step") or 0), event.get("ts"))
            return
        if kind == "val":
            cmmd = event.get("cmmd")
            if cmmd is not None and self._bar.isVisible():
                self._bar.setFormat(self._bar.format() + f" — CMMD {cmmd:.4f}")

    def _update_bar(self, cur: int, ts: float | None = None) -> None:
        total = self._total_steps
        rate = self._rate(cur, ts)
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(cur)
            self._bar.setFormat(f"step {cur}/{total} (%p%){rate}")
        else:
            self._bar.setRange(0, 0)
            self._bar.setFormat(f"step {cur}{rate}")
        if not self._bar.isVisible():
            self._bar.setVisible(True)

    def _rate(self, cur: int, ts: float | None) -> str:
        clock = ts if ts is not None else time.monotonic()
        if self._anchor is None or cur < self._anchor[1]:
            self._anchor = (clock, cur)
            return ""
        anchor_time, anchor_step = self._anchor
        steps = cur - anchor_step
        if steps <= 0:
            return ""
        spi = (clock - anchor_time) / steps
        remaining = self._total_steps - cur
        if self._total_steps <= 0 or remaining <= 0:
            return f" — {spi:.2f}s/step"
        return f" — {spi:.2f}s/step — ETA {_format_duration(remaining * spi)}"


def _format_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"
