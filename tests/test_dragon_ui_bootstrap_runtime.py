from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dragon_bootstrap_runtime_fallback_and_mode_priority() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon bootstrap runtime checks")

    script = r"""
const mod = await import('./web/static/js/ui-bootstrap.js?dragon-runtime-test');

class FakeStyle {
    constructor() { this.values = new Map(); }
    setProperty(key, value) { this.values.set(key, value); }
    removeProperty(key) { this.values.delete(key); }
}

class FakeElement {
    constructor(id) {
        this.id = id;
        this.hidden = false;
        this.dataset = {};
        this.style = new FakeStyle();
        this.attributes = new Set();
        this.childCount = 0;
    }
    hasAttribute(name) {
        if (name === 'data-dragon-ui') return Object.hasOwn(this.dataset, 'dragonUi');
        return this.attributes.has(name);
    }
    removeAttribute(name) {
        this.attributes.delete(name);
        if (name === 'data-ui-scale') delete this.dataset.uiScale;
    }
    replaceChildren() { this.childCount = 0; }
}

function installDom(mode) {
    const root = new FakeElement('dragon-root');
    const nav = new FakeElement('dragon-nav');
    const main = new FakeElement('dragon-main');
    const body = new FakeElement('body');
    body.hidden = false;
    if (mode === 'dragon') body.dataset.dragonUi = '';
    const documentElement = new FakeElement('html');
    documentElement.dataset.uiMode = mode;
    const elements = new Map([
        ['dragon-root', root],
        ['dragon-nav', nav],
        ['dragon-main', main],
    ]);
    globalThis.document = {
        body,
        documentElement,
        getElementById: (id) => elements.get(id) || null,
    };
    return { root, nav, main, body, documentElement };
}

const originalError = console.error;
console.error = () => {};

const fallbackDom = installDom('dragon');
const fallbackCalls = { classic: 0, dragon: 0, modes: [], switch: 0, tab: 0 };
await mod.bootstrapUI({
    stylesheetLoader: async (mode) => fallbackCalls.modes.push(mode),
    dragonLoader: async () => ({
        initDragonUI: async () => {
            fallbackCalls.dragon += 1;
            fallbackDom.nav.childCount = 2;
            fallbackDom.main.childCount = 3;
            fallbackDom.main.dataset.uiScale = '125';
            fallbackDom.main.style.setProperty('zoom', '1.25');
            fallbackDom.body.dataset.dragonMobileMenuOpen = '';
            fallbackDom.documentElement.dataset.dragonTheme = 'dark';
            fallbackDom.documentElement.style.setProperty('--dragon-user-scale', '1.25');
            throw new Error('simulated Dragon init failure');
        },
        destroyDragonUI: () => {},
    }),
    classicLoader: async () => ({
        startClassicUI: async () => { fallbackCalls.classic += 1; },
    }),
    modeLoader: async () => ({
        initClassicUiSwitch: () => { fallbackCalls.switch += 1; },
        activateRequestedClassicTab: () => { fallbackCalls.tab += 1; },
    }),
});

const classicDom = installDom('classic');
const classicCalls = { classic: 0, dragon: 0, modes: [] };
await mod.bootstrapUI({
    stylesheetLoader: async (mode) => classicCalls.modes.push(mode),
    dragonLoader: async () => {
        classicCalls.dragon += 1;
        return { initDragonUI: async () => {} };
    },
    classicLoader: async () => ({
        startClassicUI: async () => { classicCalls.classic += 1; },
    }),
    modeLoader: async () => ({
        initClassicUiSwitch: () => {},
        activateRequestedClassicTab: () => {},
    }),
});

console.error = originalError;
console.log(JSON.stringify({
    resolution: {
        queryClassic: mod.resolveRequestedUIMode('?ui=classic', 'dragon'),
        queryDragon: mod.resolveRequestedUIMode('?ui=dragon', 'classic'),
        storedClassic: mod.resolveRequestedUIMode('', 'classic'),
        invalidUsesDefault: mod.resolveRequestedUIMode('?ui=unknown', 'dragon'),
    },
    fallback: {
        calls: fallbackCalls,
        rootHidden: fallbackDom.root.hidden,
        navChildren: fallbackDom.nav.childCount,
        mainChildren: fallbackDom.main.childCount,
        mainScale: fallbackDom.main.dataset.uiScale || null,
        mainZoom: fallbackDom.main.style.values.get('zoom') || null,
        mobileMenu: Object.hasOwn(fallbackDom.body.dataset, 'dragonMobileMenuOpen'),
        dragonUi: Object.hasOwn(fallbackDom.body.dataset, 'dragonUi'),
        dragonTheme: Object.hasOwn(fallbackDom.documentElement.dataset, 'dragonTheme'),
        baseScale: fallbackDom.documentElement.style.values.get('--dragon-user-scale') || null,
        fallbackFlag: fallbackDom.documentElement.dataset.dragonFallback,
        uiMode: fallbackDom.documentElement.dataset.uiMode,
        boot: fallbackDom.documentElement.dataset.appBoot,
        bodyHidden: fallbackDom.body.hidden,
    },
    classic: {
        calls: classicCalls,
        uiMode: classicDom.documentElement.dataset.uiMode,
        boot: classicDom.documentElement.dataset.appBoot,
        bodyHidden: classicDom.body.hidden,
    },
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["resolution"] == {
        "queryClassic": "classic",
        "queryDragon": "dragon",
        "storedClassic": "classic",
        "invalidUsesDefault": "dragon",
    }
    assert payload["fallback"] == {
        "calls": {"classic": 1, "dragon": 1, "modes": ["dragon", "classic"], "switch": 1, "tab": 1},
        "rootHidden": True,
        "navChildren": 0,
        "mainChildren": 0,
        "mainScale": None,
        "mainZoom": None,
        "mobileMenu": False,
        "dragonUi": False,
        "dragonTheme": False,
        "baseScale": None,
        "fallbackFlag": "true",
        "uiMode": "classic",
        "boot": "classic",
        "bodyHidden": False,
    }
    assert payload["classic"] == {
        "calls": {"classic": 1, "dragon": 0, "modes": ["classic"]},
        "uiMode": "classic",
        "boot": "classic",
        "bodyHidden": False,
    }
