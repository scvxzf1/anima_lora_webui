from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / "web/static" / relative).read_text(encoding="utf-8")


def test_history_rows_reserve_metrics_and_truncate_names() -> None:
    css = _read("css/dragon/03a-dragon-history-workbench.css")
    assert "grid-template-columns:auto minmax(0,1fr) auto auto minmax(240px,auto)" in css
    assert ".dragon-history-config-task-list .dragon-dataset-preset-item{display:block;min-width:0;overflow:hidden" in css
    assert ".dragon-history-config-task-list .dragon-dataset-preset-item > span{display:block;min-width:0;overflow:hidden}" in css


def test_dataset_mobile_savebar_prioritizes_primary_action() -> None:
    css = _read("css/dragon/06-dragon-pages.css")
    assert ".dragon-dataset-savebar > div:first-child span { display: none; }" in css
    assert ".dragon-dataset-savebar-actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert '.dragon-dataset-savebar-actions [data-workspace-action="save"] { grid-column: 1 / -1;' in css


def test_dashboard_formats_epoch_timestamps_and_keeps_raw_title() -> None:
    js = _read("js/dragon-ui/pages/dashboard.js")
    assert "function formatTaskTimestamp(value)" in js
    assert "new Intl.DateTimeFormat('zh-CN'" in js
    assert 'title="${escapeHtml(rawTime)}"' in js
