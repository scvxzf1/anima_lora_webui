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
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._bar.setVisible(False)
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
            self._bar.setMaximum(tot)
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
        return f" — {spi:.2f}s/step"
