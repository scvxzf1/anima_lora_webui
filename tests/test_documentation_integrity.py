from __future__ import annotations

import re
import tomllib
import unicodedata
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
ARCHIVE_DOCS_ROOT = ROOT / "_archive" / "docs"
MAINTENANCE_DOCS = (
    ROOT / "AGENTS.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "networks" / "CLAUDE.md",
    ROOT / "custom_nodes" / "comfyui-hydralora" / "CLAUDE.md",
)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)(?:\s+#+\s*)?$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _markdown_files() -> list[Path]:
    return [
        ROOT / "README.md",
        *MAINTENANCE_DOCS,
        *sorted(DOCS_ROOT.rglob("*.md")),
        *sorted(ARCHIVE_DOCS_ROOT.rglob("*.md")),
    ]


def _content_lines(path: Path):
    fenced = False
    marker = ""
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        match = FENCE_RE.match(line)
        if match:
            current = match.group(1)
            if not fenced:
                fenced = True
                marker = current
            elif current == marker:
                fenced = False
                marker = ""
            continue
        if not fenced:
            yield line_number, line


def _local_link_targets(path: Path):
    for line_number, line in _content_lines(path):
        for match in LINK_RE.finditer(line):
            raw = match.group(1).strip()
            raw = raw.split(' "', 1)[0].split(" '", 1)[0]
            if not raw or raw.startswith(
                ("#", "http://", "https://", "mailto:", "data:", "javascript:")
            ):
                continue
            path_part, separator, fragment = raw.partition("#")
            target = (path.parent / unquote(path_part)).resolve()
            yield line_number, raw, target, unquote(fragment) if separator else ""


def _github_slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>]*>", "", text).strip().lower()
    output: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character in "_-":
            output.append(character)
        elif character.isspace():
            output.append("-")
        elif category[0] in "LN" or category in {"Mn", "Mc"}:
            output.append(character)
    return "".join(output)


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for _, line in _content_lines(path):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = _github_slug(match.group(1))
        duplicate_index = counts.get(base, 0)
        counts[base] = duplicate_index + 1
        anchors.add(base if duplicate_index == 0 else f"{base}-{duplicate_index}")
    return anchors


def test_markdown_fences_are_balanced():
    failures: list[str] = []
    for path in _markdown_files():
        marker = ""
        start_line = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            match = FENCE_RE.match(line)
            if not match:
                continue
            current = match.group(1)
            if not marker:
                marker = current
                start_line = line_number
            elif current == marker:
                marker = ""
                start_line = 0
        if marker:
            failures.append(f"{path.relative_to(ROOT)}:{start_line}")
    assert not failures, "unclosed Markdown fences:\n" + "\n".join(failures)


def test_local_markdown_links_and_anchors_exist():
    missing: list[str] = []
    bad_anchors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for path in _markdown_files():
        for line_number, raw, target, fragment in _local_link_targets(path):
            if not target.exists():
                missing.append(f"{path.relative_to(ROOT)}:{line_number}: {raw}")
                continue
            if fragment and target.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(target, _anchors(target))
                fragment = fragment.split(" ", 1)[0]
                if fragment not in anchors:
                    bad_anchors.append(
                        f"{path.relative_to(ROOT)}:{line_number}: {raw}"
                    )
    assert not missing, "missing local Markdown targets:\n" + "\n".join(missing)
    assert not bad_anchors, "missing Markdown anchors:\n" + "\n".join(bad_anchors)


def test_every_document_is_reachable_from_docs_index():
    documents = {
        path.resolve()
        for root in (DOCS_ROOT, ARCHIVE_DOCS_ROOT)
        for path in root.rglob("*.md")
    }
    edges: dict[Path, set[Path]] = {path: set() for path in documents}
    for path in documents:
        for _, _, target, _ in _local_link_targets(path):
            if target in documents:
                edges[path].add(target)

    start = (DOCS_ROOT / "README.md").resolve()
    reached: set[Path] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(edges[current] - reached)

    unreachable = sorted(path.relative_to(ROOT) for path in documents - reached)
    assert not unreachable, "unreachable documentation:\n" + "\n".join(
        map(str, unreachable)
    )


def test_section_indexes_list_every_document():
    failures: list[str] = []
    for section in sorted(path for path in DOCS_ROOT.iterdir() if path.is_dir()):
        index = section / "README.md"
        if not index.exists():
            continue
        linked: set[Path] = set()
        for _, _, target, _ in _local_link_targets(index):
            try:
                target.relative_to(section.resolve())
            except ValueError:
                continue
            if target.suffix.lower() == ".md":
                linked.add(target)
        actual = {path.resolve() for path in section.rglob("*.md") if path != index}
        for path in sorted(actual - linked):
            failures.append(f"{index.relative_to(ROOT)} missing {path.relative_to(ROOT)}")
    assert not failures, "incomplete documentation indexes:\n" + "\n".join(
        failures
    )


def test_lifecycle_sensitive_sections_have_status_labels():
    failures: list[str] = []
    markers = ("状态：", "Status:", "Current status:", "> Current status:")
    for section_name in ("experimental", "findings", "proposal"):
        for path in (DOCS_ROOT / section_name).rglob("*.md"):
            header = "\n".join(
                path.read_text(encoding="utf-8", errors="replace").splitlines()[:25]
            )
            if not any(marker in header for marker in markers):
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, "missing lifecycle status labels:\n" + "\n".join(failures)


def test_current_optimization_document_matches_live_config_surfaces():
    document = (DOCS_ROOT / "optimization-configs-current.md").read_text(
        encoding="utf-8"
    )
    base = tomllib.loads((ROOT / "configs/base.toml").read_text(encoding="utf-8"))
    presets = tomllib.loads(
        (ROOT / "configs/presets.toml").read_text(encoding="utf-8")
    )

    custom_down = str(bool(base["use_custom_down_autograd"])).lower()
    custom_down_line = next(
        line for line in document.splitlines() if line.startswith("| `use_custom_down_autograd`")
    )
    assert f"base `{custom_down}`" in custom_down_line

    preset_line = next(
        line for line in document.splitlines() if line.startswith("| `preset` |")
    )
    assert not [name for name in presets if f"`{name}`" not in preset_line]

    variant_line = next(
        line for line in document.splitlines() if line.startswith("| 自包含变体文件 |")
    )
    variants = {path.stem for path in (ROOT / "configs/gui-methods").glob("*.toml")}
    assert not [name for name in sorted(variants) if f"`{name}`" not in variant_line]

    router_line = next(
        line for line in document.splitlines() if line.startswith("| `router_source` |")
    )
    config_source = (ROOT / "networks/lora_anima/config.py").read_text(
        encoding="utf-8"
    )
    literal = re.search(r"RouterSource\s*=\s*Literal\[([^\]]+)\]", config_source)
    assert literal is not None
    router_sources = set(re.findall(r'["\']([^"\']+)["\']', literal.group(1)))
    assert not [name for name in sorted(router_sources) if f"`{name}`" not in router_line]


def test_configuration_and_training_docs_match_current_defaults():
    external = (DOCS_ROOT / "configuration/external-configs.md").read_text(
        encoding="utf-8"
    )
    priority = [
        external.index(".anima-webui-settings.toml [paths].configs_root"),
        external.index("`ANIMA_CONFIGS_ROOT` 环境变量"),
        external.index("默认 `configs/` 目录"),
    ]
    assert priority == sorted(priority)

    base = tomllib.loads((ROOT / "configs/base.toml").read_text(encoding="utf-8"))
    training = (DOCS_ROOT / "guidelines/training.md").read_text(encoding="utf-8")
    dataset = base["datasets"][0]
    if not base["use_cmmd"] and dataset["validation_split_num"] == 0:
        assert "Validation is **disabled by default**" in training
    assert '"crossattn_emb"' in training


def test_archived_proposals_are_listed_in_archive_indexes() -> None:
    archived = sorted(
        path.name
        for path in (ARCHIVE_DOCS_ROOT / "proposal").glob("*.md")
        if path.name != "README.md"
    )
    archive_index = (DOCS_ROOT / "archive-index.md").read_text(encoding="utf-8")
    proposal_index = (ARCHIVE_DOCS_ROOT / "proposal" / "README.md").read_text(
        encoding="utf-8"
    )

    assert archived
    assert [name for name in archived if name not in archive_index] == []
    assert [name for name in archived if name not in proposal_index] == []
