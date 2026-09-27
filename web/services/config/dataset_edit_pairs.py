"""Translate editor-only Qwen edit source rows to executable dataset rows."""

from __future__ import annotations

from typing import Any


LAYOUT_KEY = "webui_edit_layout"
ROLES = {"normal", "before", "after"}


def project_edit_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not any(str(row.get("edit_role") or "normal") != "normal" for row in rows):
        return rows, []

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        role = str(row.get("edit_role") or "normal")
        pair_id = str(row.get("edit_pair_id") or "").strip()
        if role not in ROLES:
            raise ValueError(f"未知编辑子集角色: {role}")
        if role == "normal":
            raise ValueError("编辑 LoRA 预设不能混用普通数据子集")
        if not pair_id:
            raise ValueError("编辑前/编辑后子集必须填写配对名称")
        if row.get("is_reg"):
            raise ValueError("编辑 LoRA 不支持正则数据子集")
        pair = pairs.setdefault(pair_id, {})
        if role in pair:
            raise ValueError(f"配对 {pair_id} 有重复的编辑{'前' if role == 'before' else '后'}子集")
        pair[role] = row

    for pair_id, pair in pairs.items():
        if set(pair) != {"before", "after"}:
            raise ValueError(f"配对 {pair_id} 必须各有一个编辑前和编辑后子集")

    runtime_rows: list[dict[str, Any]] = []
    layout: list[dict[str, Any]] = []
    for row in rows:
        role = str(row["edit_role"])
        pair_id = str(row["edit_pair_id"]).strip()
        if role == "before":
            source = str(row.get("source_dir") or "").strip()
            if not source:
                raise ValueError(f"配对 {pair_id} 的编辑前图片目录不能为空")
            layout.append({"role": role, "pair_id": pair_id, "source_dir": source})
            continue
        target = dict(row)
        target.pop("edit_role", None)
        target.pop("edit_pair_id", None)
        target["reference_image_dir"] = str(pairs[pair_id]["before"]["source_dir"]).strip()
        runtime_rows.append(target)
        layout.append({"role": role, "pair_id": pair_id, "runtime_index": len(runtime_rows) - 1})
    return runtime_rows, layout


def restore_edit_rows(rows: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    general = data.get("general") or {}
    attrs = (general.get("custom_attributes") or {}) if isinstance(general, dict) else {}
    layout = attrs.get(LAYOUT_KEY) if isinstance(attrs, dict) else None
    if not isinstance(layout, list) or not layout:
        return rows
    restored: list[dict[str, Any]] = []
    used: set[int] = set()
    for item in layout:
        if not isinstance(item, dict):
            raise ValueError("编辑子集布局损坏")
        role = str(item.get("role") or "")
        pair_id = str(item.get("pair_id") or "")
        if role == "before":
            source = str(item.get("source_dir") or "")
            restored.append({"edit_role": role, "edit_pair_id": pair_id,
                             "source_dir": source, "image_dir": source})
        elif role == "after":
            index = item.get("runtime_index")
            if not isinstance(index, int) or index in used or not 0 <= index < len(rows):
                raise ValueError("编辑子集布局索引损坏")
            used.add(index)
            restored.append({**rows[index], "edit_role": role, "edit_pair_id": pair_id})
        else:
            raise ValueError("编辑子集布局角色损坏")
    if used != set(range(len(rows))):
        raise ValueError("编辑子集布局与训练子集不匹配")
    projected, _ = project_edit_rows(restored)
    if any(
        str(projected[index].get("reference_image_dir") or "")
        != str(rows[index].get("reference_image_dir") or "")
        for index in range(len(rows))
    ):
        raise ValueError("编辑子集布局与训练参考图目录不一致")
    return restored
