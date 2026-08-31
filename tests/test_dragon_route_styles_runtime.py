from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import STATIC_DIR


def test_route_styles_swap_atomically_and_preserve_active_styles_on_error() -> None:
    module_uri = (STATIC_DIR / "js/dragon-ui/route-styles.js").resolve().as_uri()
    script = f"""
const children = [];

function createLink() {{
  const listeners = new Map();
  return {{
    dataset: {{}},
    rel: '',
    href: '',
    addEventListener(type, callback, options = {{}}) {{
      const entries = listeners.get(type) || [];
      entries.push({{ callback, once: Boolean(options.once) }});
      listeners.set(type, entries);
    }},
    emit(type) {{
      const entries = listeners.get(type) || [];
      for (const entry of entries) entry.callback();
      listeners.set(type, entries.filter((entry) => !entry.once));
    }},
    remove() {{
      const index = children.indexOf(this);
      if (index >= 0) children.splice(index, 1);
    }},
  }};
}}

globalThis.document = {{
  createElement(tag) {{
    if (tag !== 'link') throw new Error(`unexpected element: ${{tag}}`);
    return createLink();
  }},
  head: {{ appendChild(link) {{ children.push(link); }} }},
  querySelectorAll(selector) {{
    if (selector !== 'link[data-dragon-route-style]') return [];
    return children.filter((link) => link.dataset.dragonRouteStyle === 'true');
  }},
}};

const styles = await import({json.dumps(module_uri + '?runtime-test')});

const dashboardLoad = styles.ensureDragonRouteStyles('dashboard');
const dashboardCandidates = [...children];
const dashboardPendingCount = children.length;
dashboardCandidates.forEach((link) => link.emit('load'));
await dashboardLoad;
const dashboardActiveCount = children.length;

await styles.ensureDragonRouteStyles('dashboard');
const dashboardRepeatCount = children.length;

const configLoad = styles.ensureDragonRouteStyles('config');
const configCandidates = children.filter((link) => !dashboardCandidates.includes(link));
const configPendingCount = children.length;
const dashboardStillMounted = dashboardCandidates.every((link) => children.includes(link));
configCandidates.forEach((link) => link.emit('load'));
await configLoad;
const configActive = [...children];
const configActiveCount = configActive.length;
const dashboardRemoved = dashboardCandidates.every((link) => !children.includes(link));

const datasetLoad = styles.ensureDragonRouteStyles('dataset');
const datasetCandidates = children.filter((link) => !configActive.includes(link));
datasetCandidates[0].emit('error');
let datasetRejected = false;
try {{
  await datasetLoad;
}} catch {{
  datasetRejected = true;
}}
const configPreservedAfterError = configActive.every((link) => children.includes(link));
const failedCandidatesRemoved = datasetCandidates.every((link) => !children.includes(link));

styles.clearDragonRouteStyles();
console.log(JSON.stringify({{
  dashboardPendingCount,
  dashboardActiveCount,
  dashboardRepeatCount,
  configPendingCount,
  dashboardStillMounted,
  configActiveCount,
  dashboardRemoved,
  datasetRejected,
  configPreservedAfterError,
  failedCandidatesRemoved,
  remainingAfterClear: children.length,
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "dashboardPendingCount": 3,
        "dashboardActiveCount": 3,
        "dashboardRepeatCount": 3,
        "configPendingCount": 10,
        "dashboardStillMounted": True,
        "configActiveCount": 7,
        "dashboardRemoved": True,
        "datasetRejected": True,
        "configPreservedAfterError": True,
        "failedCandidatesRemoved": True,
        "remainingAfterClear": 0,
    }
