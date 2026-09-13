"""Behavioral coverage for the compact workspace, independent of visual tokens."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "web/static/js/dragon-ui/pages"


def _run(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("node is required for workspace runtime checks")
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=20,
    )
    return json.loads(result.stdout)


def _uri(name: str) -> str:
    return json.dumps((PAGES / name).as_uri())


DOM = """
class Element {
  constructor(dataset = {}) {
    this.dataset = dataset; this.hidden = false; this.value = '';
    this.textContent = ''; this.attrs = {}; this.listeners = new Map();
    this.one = {}; this.many = {}; this.scrollTop = 0;
    this.classList = { contains: () => true };
  }
  querySelector(key) { return this.one[key] || null; }
  querySelectorAll(key) { return this.many[key] || []; }
  setAttribute(key, value) { this.attrs[key] = value; }
  removeAttribute(key) { delete this.attrs[key]; }
  addEventListener(key, fn) { this.listeners.set(key, fn); }
  removeEventListener(key, fn) { if(this.listeners.get(key) === fn) this.listeners.delete(key); }
  getBoundingClientRect() { return {top: 100, bottom: 600}; }
  scrollTo({top}) { this.scrollTop = top; }
  closest() { return null; }
}
globalThis.window = {
  scrollY: 0, innerHeight: 844,
  scrollTo({top}) { this.scrollY = top; },
  setTimeout, clearTimeout,
  matchMedia: () => ({matches: false}),
};
globalThis.document = {documentElement: {dataset: {}}, body: {dataset: {}}};
globalThis.getComputedStyle = () => ({overflowY: 'auto', scrollPaddingTop: '16px'});
"""


def test_workspace_preferences_migrate_without_removing_saved_choices() -> None:
    result = _run(f"""
const storage = new Map([['anima_dragon_config_ui', JSON.stringify({{
  viewMode: 'grouped', presetCollapsed: true, visibilityLevel: 'newcomer', bilingual: true,
}})]]);
globalThis.localStorage = {{getItem: k => storage.get(k), setItem: (k,v) => storage.set(k,v)}};
const prefs = await import({_uri('config-ui-preferences.js')});
const initial = [prefs.preferredConfigSubId(null, {{id:'training-config'}}),
  prefs.presetLibraryCollapsed(), prefs.preferredConfigVisibilityLevel(), prefs.preferredConfigBilingual()];
prefs.persistPresetLibraryCollapsed(true);
prefs.persistConfigVisibilityLevel('newcomer');
const explicit = [prefs.presetLibraryCollapsed(), prefs.preferredConfigVisibilityLevel(),
  prefs.preferredConfigSubId('common', {{id:'training-config'}})];
console.log(JSON.stringify({{initial, explicit, saved: JSON.parse(storage.get('anima_dragon_config_ui'))}}));
""")
    assert result["initial"] == ["all", False, "all", True]
    assert result["explicit"] == [True, "newcomer", "common"]
    assert result["saved"]["bilingual"] is True
    assert result["saved"]["workspaceVersion"] == 1
    assert result["saved"]["fieldScopeVersion"] == 1


def test_search_bypasses_tiers_not_applicability_and_restores_browsing() -> None:
    result = _run(DOM + f"""
const mod = await import({_uri('config-field-filter.js')});
const root = new Element(), input = new Element(), output = new Element(), canvas = new Element();
const empty = new Element(), nav = new Element({{configTagFilter:'input'}});
const fields = ['learning_rate', 'advanced_path', 'lokr_factor'].map((key, i) => {{
  const field = new Element({{searchText:key, configVisibilityLevel:i ? 'advanced' : 'newcomer',
    configPresentationVisible:i === 2 ? 'false' : 'true'}});
  const control = new Element({{key}}); control.value = i === 1 ? '/models/example.safetensors' : '16';
  field.one['[data-key]'] = control;
  return field;
}});
const stages = fields.map(field => {{
  const group = new Element();
  group.many['.dragon-field, .dragon-config-dataset-card'] = [field];
  group.one['[data-config-stage-count]'] = new Element();
  group.one['[data-config-stage-empty]'] = new Element();
  return group;
}});
root.one = {{'[data-config-field-search]':input, '[data-config-field-filter-count]':output,
  '#dragon-config-fields':canvas}};
root.many['[data-config-tag-filter]'] = [nav];
canvas.one = {{'[data-config-filter-empty]':empty, '[data-config-section="input"]':stages[0]}};
canvas.many = {{'.dragon-field, .dragon-config-dataset-card':fields,
  '[data-config-filter-group]':stages, '[data-config-cluster-group]':stages}};
const label = new Element(), changed = new Element();
const state = {{configVisibilityLevel:'newcomer', dirtyKeys:new Set(['lokr_factor']), dirty:true,
  dirtyBindings:{{fields:new Map(), count:new Element(), changedOnly:changed, changedOnlyLabel:label}}}};
const cleanup = mod.bindConfigFieldFilter(root,state);
const initial = fields.map(f => !f.hidden);
canvas.scrollTop = 250;
input.value = 'example.safetensors'; state.filterUpdate();
const valueSearch = {{fields:fields.map(f => !f.hidden), stages:stages.map(s=>!s.hidden), top:canvas.scrollTop}};
input.value = 'lokr_factor'; state.filterUpdate();
const hiddenSearch = {{visible:!fields[2].hidden, empty:!empty.hidden, count:output.textContent}};
input.value = ''; state.filterUpdate();
const restored = canvas.scrollTop;
state.showChangedOnly = true; state.filterUpdate();
const hiddenDirty = fields.map(f=>!f.hidden);
nav.listeners.get('click')();
const navigated = {{changed:state.showChangedOnly, label:label.textContent, fields:fields.map(f=>!f.hidden)}};
cleanup();
console.log(JSON.stringify({{initial,valueSearch,hiddenSearch,restored,hiddenDirty,navigated,
  listeners:input.listeners.size + nav.listeners.size, cleaned:state.filterUpdate === null}}));
""")
    assert result["initial"] == [True, False, False]
    assert result["valueSearch"] == {"fields": [False, True, False], "stages": [False, True, False], "top": 0}
    assert result["hiddenSearch"]["visible"] is False
    assert result["hiddenSearch"]["empty"] is True
    assert "隐藏匹配 1" in result["hiddenSearch"]["count"]
    assert result["restored"] == 250
    assert result["hiddenDirty"] == [False, False, True]
    assert result["navigated"] == {"changed": False, "label": "查看修改", "fields": [True, False, False]}
    assert result["listeners"] == 0
    assert result["cleaned"] is True


def test_library_search_restores_group_collapse_across_renders() -> None:
    result = _run(DOM + f"""
const mod = await import({_uri('config-library-navigation.js')});
const state = {{}};
function render() {{
  const root = new Element(), search = new Element(), empty = new Element();
  const group = new Element({{trainingPresetGroup:'saved'}}), button = new Element(), list = new Element();
  button.textContent = 'Saved';
  const rows = ['alpha.toml', 'beta.toml'].map(text => {{const row=new Element();row.textContent=text;return row;}});
  root.one = {{'[data-training-preset-search]':search, '[data-training-preset-no-results]':empty}};
  root.many['[data-training-preset-group]']=[group];
  group.one = {{'[data-training-group-toggle]':button, '.dragon-training-preset-list':list}};
  group.many['[data-training-preset-row]']=rows;
  mod.bindConfigLibraryNavigation(root,state);
  return {{root,search,empty,group,button,list,rows}};
}}
let view = render();
view.button.listeners.get('click')();
const collapsed = view.list.hidden;
view.search.value = 'beta'; view.search.listeners.get('input')();
const searching = [view.list.hidden, ...view.rows.map(row=>row.hidden)];
view = render();
const rerendered = [view.search.value, ...view.rows.map(row=>row.hidden)];
view.search.value = ''; view.search.listeners.get('input')();
const restored = view.list.hidden;
view.search.value = 'missing'; view.search.listeners.get('input')();
console.log(JSON.stringify({{collapsed,searching,rerendered,restored,empty:!view.empty.hidden,groupHidden:view.group.hidden}}));
""")
    assert result == {
        "collapsed": True, "searching": [False, True, False],
        "rerendered": ["beta", True, False], "restored": True,
        "empty": True, "groupHidden": True,
    }


def test_scroll_switches_between_desktop_canvas_and_mobile_page() -> None:
    result = _run(DOM + f"""
const mod = await import({_uri('config-workspace-scroll.js')});
const canvas = new Element(), target = new Element();
canvas.scrollTop = 200;
target.getBoundingClientRect = () => ({{top:400,bottom:440}});
mod.scrollConfigCanvasTo(canvas,target,'instant');
const desktop = canvas.scrollTop;
globalThis.getComputedStyle = () => ({{overflowY:'visible',scrollPaddingTop:'16px',top:'52px'}});
const detail = new Element(), toolbar = new Element(), footer = new Element();
toolbar.getBoundingClientRect = () => ({{top:250,bottom:350,height:100}});
footer.getBoundingClientRect = () => ({{top:780}});
detail.one = {{'.dragon-config-all-toolbar':toolbar, '.dragon-config-all-footer':footer}};
canvas.closest = () => detail;
window.scrollY = 300;
mod.scrollConfigCanvasTo(canvas,target,'instant');
const mobile = window.scrollY;
mod.restoreConfigScroll(canvas,300);
console.log(JSON.stringify({{desktop,mobile,restored:window.scrollY,canvasUnchanged:canvas.scrollTop}}));
""")
    assert result == {"desktop": 484, "mobile": 532, "restored": 300, "canvasUnchanged": 484}
