# 数据集跨分组拖动修复（2026-07-27）

状态：已完成  
适用版本：当前 `main`  
入口命令：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py::test_file_group_drag_matches_history_style_same_list_reorder \
  tests/test_training_frontend_config_ui.py::test_move_file_near_list_inserts_across_groups \
  tests/test_web_config_file_groups.py::test_place_dataset_file_cross_group_with_order_list \
  tests/test_web_config_file_groups.py::test_place_config_file_in_group_accepts_full_order_list \
  tests/test_web_config_file_groups.py::test_place_config_file_in_group_same_group_uses_anchor_before_after \
  -q
```

相关代码：

- `web/static/js/features/toml-manager/file-group-drag-core.js`
- `web/static/js/features/toml-manager/file-group-drag-targets.js`
- `web/services/config/file_group_ops_mutate.py`（后端已支持，本次未改）

---

## 1. 现象

数据集预设界面：

- 同组内拖动排序：正常
- 拖到另一个分组落地：无效果（视觉 drop 高亮可能出现，但文件不迁移）

TOML 文件分组与数据集预设共用同一套 file-group drag 实现。

## 2. 根因

同组 reorder 路径通过 `resolveFileGroupSameListDrop` 读取目标 DOM 的完整 `order`，再经
`moveFileNearList(currentOrder, source, anchor, position)` 生成 `nextOrder`，提交给
`POST /api/config/file-groups/place`。

跨组时：

1. 目标 list 的 DOM order **不含** source
2. 旧实现：

   ```js
   if (!source || !out.includes(source)) return out;
   ```

   source 不在 list → 直接返回原 list
3. `unchanged = nextOrder === currentOrder` 为 true → drop handler early-return
4. 后端 place 从未被调用 → 静默失败

后端 `place_config_file_in_group(..., order=)` 本身已接受「order 含尚未在目标组的文件」，
问题只在前端 order 构造。

## 3. 修复

`moveFileNearList`：

- 去掉 `!out.includes(source)` early-return
- source 已在 list：先 splice 再插（同组行为不变）
- source 不在 list：直接插到 anchor 的 before/after（跨组插入）

`resolveFileGroupSameListDrop` 的 `unchanged` 判断仍保留：跨组时 `nextOrder` 比
`currentOrder` 多一项 source，不会误判 no-op。

## 4. 验证

| 用例 | 结果 |
|------|------|
| 同组 reorder 契约（禁止 early-return） | pass |
| `moveFileNearList` 同组 / 跨组 / 空目标 / 真 no-op | pass |
| 后端 place 带完整 order 跨组迁移 dataset 文件 | pass |
| 既有 full-order / anchor before-after | pass |

`node --check` 通过；未提交 audit 临时文件（`tmp/`、`eslint.audit.config.mjs` 等）。
