from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import REPO_ROOT


def test_live_log_search_batches_renders_and_cleans_up() -> None:
    script = r"""
const timers = new Map();
let nextTimer = 1;
globalThis.window = {
  setTimeout(callback, delay) {
    const id = nextTimer++;
    timers.set(id, { callback, delay });
    return id;
  },
  clearTimeout(id) { timers.delete(id); },
};

function eventNode(extra = {}) {
  const listeners = new Map();
  return {
    dataset: {},
    title: '',
    ...extra,
    addEventListener(type, handler) { listeners.set(type, handler); },
    removeEventListener(type, handler) {
      if (listeners.get(type) === handler) listeners.delete(type);
    },
    emit(type) { listeners.get(type)?.({ currentTarget: this }); },
    listenerCount() { return listeners.size; },
  };
}

let htmlWrites = 0;
const search = eventNode({ value: '' });
const container = eventNode({ scrollTop: 0, scrollHeight: 100, _html: '' });
Object.defineProperty(container, 'innerHTML', {
  get() { return this._html; },
  set(value) { this._html = value; htmlWrites += 1; },
});
const feedback = eventNode({ textContent: '' });
const count = eventNode({ textContent: '' });
const visibleCount = eventNode({ textContent: '' });
const pauseLabel = eventNode({ textContent: '' });
const pause = eventNode({
  querySelector() { return pauseLabel; },
  getAttribute(name) { return this[name] ?? null; },
  setAttribute(name, value) { this[name] = value; },
});
const actions = {
  copy: eventNode(), download: eventNode(), pause, clear: eventNode(),
};
const nodes = new Map([
  ['[data-live-log-search]', search],
  ['[data-live-log]', container],
  ['[data-live-log-feedback]', feedback],
  ['[data-live-log-count]', count],
  ['[data-live-log-visible-count]', visibleCount],
  ['[data-live-log-action="copy"]', actions.copy],
  ['[data-live-log-action="download"]', actions.download],
  ['[data-live-log-action="pause"]', actions.pause],
  ['[data-live-log-action="clear"]', actions.clear],
]);
const root = { querySelector(selector) { return nodes.get(selector) || null; } };

const mod = await import('./web/static/js/dragon-ui/pages/live-training-log-tools.js?runtime-test');
const bindings = mod.createLiveLogBindings(root);
const model = {
  logs: Array.from({ length: 300 }, (_, id) => ({ id: id + 1, message: `line ${id}` })),
  logQuery: '', logClearBeforeId: 0, autoScroll: false,
};
const cleanup = mod.bindLiveLogTools(bindings, model, () => {});

for (let index = 0; index < 20; index += 1) {
  search.value = `line ${index}`;
  search.emit('input');
}
const beforeFlush = {
  timers: timers.size,
  delays: [...timers.values()].map((timer) => timer.delay),
  htmlWrites,
  query: model.logQuery,
};
const timer = [...timers.values()][0];
timers.clear();
timer.callback();
const afterFlush = { timers: timers.size, htmlWrites };

search.value = 'pending';
search.emit('input');
cleanup();
const listenerCount = [search, ...Object.values(actions)]
  .reduce((sum, node) => sum + node.listenerCount(), 0);
console.log(JSON.stringify({ beforeFlush, afterFlush, afterCleanup: { timers: timers.size, listenerCount } }));
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
    assert payload["beforeFlush"] == {
        "timers": 1,
        "delays": [100],
        "htmlWrites": 0,
        "query": "line 19",
    }
    assert payload["afterFlush"] == {"timers": 0, "htmlWrites": 1}
    assert payload["afterCleanup"] == {"timers": 0, "listenerCount": 0}
