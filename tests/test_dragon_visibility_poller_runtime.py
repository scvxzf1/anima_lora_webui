from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_visibility_poller_pauses_hidden_work_and_resumes_once() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon poller runtime checks")

    script = r"""
const timers = new Map();
let nextTimer = 1;
let visibilityHandler = null;
globalThis.window = {
    setTimeout(callback, delay) {
        const id = nextTimer++;
        timers.set(id, { callback, delay });
        return id;
    },
    clearTimeout(id) { timers.delete(id); },
};
globalThis.document = {
    hidden: false,
    addEventListener(name, handler) {
        if (name === 'visibilitychange') visibilityHandler = handler;
    },
    removeEventListener(name, handler) {
        if (name === 'visibilitychange' && visibilityHandler === handler) visibilityHandler = null;
    },
};

const { createVisibilityPoller } = await import(
    './web/static/js/dragon-ui/visibility-poller.js?runtime-test'
);
const calls = [];
const poller = createVisibilityPoller({
    poll: async () => { calls.push('poll'); },
    delay: () => 5000,
});

poller.start();
const firstSchedule = [...timers.values()].map((item) => item.delay);
const firstTimer = [...timers.values()][0];
await firstTimer.callback();
const afterPoll = { calls: [...calls], timers: timers.size };

document.hidden = true;
visibilityHandler();
const whileHidden = timers.size;
document.hidden = false;
visibilityHandler();
await Promise.resolve();
await Promise.resolve();
const afterResume = { calls: [...calls], timers: timers.size };

poller.stop();
console.log(JSON.stringify({
    firstSchedule,
    afterPoll,
    whileHidden,
    afterResume,
    stopped: { timers: timers.size, listener: visibilityHandler },
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["firstSchedule"] == [5000]
    assert payload["afterPoll"] == {"calls": ["poll"], "timers": 1}
    assert payload["whileHidden"] == 0
    assert payload["afterResume"]["calls"] == ["poll", "poll"]
    assert payload["stopped"] == {"timers": 0, "listener": None}
