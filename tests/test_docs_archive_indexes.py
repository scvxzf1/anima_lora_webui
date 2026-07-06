from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_archived_proposals_are_listed_in_archive_indexes() -> None:
    archived = sorted(
        path.name
        for path in (ROOT / "_archive" / "docs" / "proposal").glob("*.md")
        if path.name != "README.md"
    )
    archive_index = (ROOT / "docs" / "archive-index.md").read_text(encoding="utf-8")
    proposal_index = (
        ROOT / "_archive" / "docs" / "proposal" / "README.md"
    ).read_text(encoding="utf-8")

    assert archived
    assert [name for name in archived if name not in archive_index] == []
    assert [name for name in archived if name not in proposal_index] == []
