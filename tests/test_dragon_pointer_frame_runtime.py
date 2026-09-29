from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import REPO_ROOT


def test_pointer_moves_are_coalesced_to_the_latest_frame() -> None:
    script = r"""
let frameCallback = null;
let frameCount = 0;
let cancelled = 0;
const listeners = new Map();
const view = {
  PointerEvent: undefined,
  requestAnimationFrame(callback) { frameCount += 1; frameCallback = callback; return frameCount; },
  cancelAnimationFrame() { cancelled += 1; frameCallback = null; },
};
const target = {
  ownerDocument: { defaultView: view },
  addEventListener(name, callback) { listeners.set(name, callback); },
  removeEventListener(name, callback) {
    if (listeners.get(name) === callback) listeners.delete(name);
  },
};
const received = [];
const mod = await import('./web/static/js/dragon-ui/pointer-frame.js?runtime-test');
const cleanup = mod.bindLatestPointerMove(target, (event) => received.push([event.clientX, event.clientY]));
for (let index = 0; index < 100; index += 1) {
  listeners.get('mousemove')({ clientX: index, clientY: index * 2, target });
}
const beforeFlush = { frameCount, received: received.length, listener: listeners.has('mousemove') };
frameCallback();
const afterFlush = { received, listener: listeners.has('mousemove') };
listeners.get('mousemove')({ clientX: 200, clientY: 300, target });
cleanup();
console.log(JSON.stringify({
  beforeFlush,
  afterFlush,
  afterCleanup: { listener: listeners.has('mousemove'), cancelled, frameCallback: Boolean(frameCallback) },
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
    assert payload["beforeFlush"] == {"frameCount": 1, "received": 0, "listener": True}
    assert payload["afterFlush"] == {"received": [[99, 198]], "listener": True}
    assert payload["afterCleanup"] == {"listener": False, "cancelled": 1, "frameCallback": False}
