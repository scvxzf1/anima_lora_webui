from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "web/static/js/features/config-form/group-entry.js"
).read_text(encoding="utf-8")


def _section(start: str, end: str) -> str:
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


def test_category_switch_preserves_search_and_scopes_results() -> None:
    select_category = _section(
        "export function selectConfigCategory", "function scrollConfigFormContentToTop"
    )

    assert "configFormState.search = '';" not in select_category
    assert "categories.filter((category) => category.id === activeCategory)" in SOURCE
    assert "createConfigFormControls(scopedGroups, renderedGroups, searchText)" in SOURCE
    assert "categoryId === activeCategory && !searchText" not in SOURCE


def test_search_filters_individual_fields_and_hides_auxiliary_panels() -> None:
    filter_group = _section("function filterConfigGroupEntry", "function configFieldMatchesSearch")
    create_group = _section("function createGroup", "return section;")

    assert "group.fields.filter" in filter_group
    assert "groupMatched" not in filter_group
    assert "const filtering = Boolean(searchText);" in create_group
    assert "!filtering && extraClass === 'config-group-model'" in create_group
    assert "!filtering && extraClass === 'config-group-data'" in create_group
    assert "!filtering && extraClass === 'config-group-resource'" in create_group
    assert "!filtering && extraClass === 'config-group-no-dataset-regularization'" in create_group
    assert "!filtering && extraClass === 'config-group-steps'" in create_group
