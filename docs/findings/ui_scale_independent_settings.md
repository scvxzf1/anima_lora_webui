# WebUI 独立界面比例落地记录

## 背景

当前 WebUI 只有一个全局 `ui_scale`，会统一作用到整个页面。目标是把它扩展成：

- 一个默认比例
- 顶层 6 个页面可独立覆盖
- 历史详情 5 个页签可独立覆盖
- 全部走现有全局设置链路持久化

## 阶段 1：扩展全局设置 schema 与持久化

- 目标：后端先支持“默认比例 + 可选覆盖比例”的保存、读取和清除。
- 改动：
  - `settings_service.py` 新增 11 个可选覆盖 key。
  - 默认值只保留 `ui_scale = 100`；覆盖项默认返回空字符串，表示“跟随默认”。
  - 保存逻辑支持写入覆盖值，也支持在传入空值时清除覆盖 key。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_ui_scale_settings.py`
  - 结果：`4 passed in 0.07s`
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_preview_service.py`
  - 结果：`21 passed in 5.74s`

## 阶段 2：全局设置界面与前端读写

- 目标：把“默认比例 + 独立覆盖”的设置项真正接到 `全局设置 -> 界面设置` 表单与保存链路。
- 改动：
  - `defaults.js` 中新增顶层 6 页与历史详情 5 页签的字段 metadata。
  - `index.html` 的“界面设置”区块新增两组独立比例表单。
  - `30-preview-settings-dialogs.css` 新增覆盖项布局与跟随默认状态样式。
  - `26-load-global-settings.js` 新增：
    - `resolveGlobalUIScaleDefaultValue`
    - `syncGlobalUIScaleOverrideField`
    - `syncAllGlobalUIScaleOverrideFields`
    - `applyGlobalUIScaleOverrideInputs`
    - `collectGlobalUIScaleOverridePayload`
  - `36-setup-event-listeners.js` 新增默认比例输入与“跟随默认”复选框的同步事件。
- 交互结果：
  - 默认比例仍由 `ui_scale` 管理。
  - 每个独立覆盖项都可在“跟随默认”和“单独输入百分比”之间切换。
  - 保存时，跟随默认会回写空字符串，由后端清除覆盖 key。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py`
  - 结果：`51 passed in 15.78s`

## 阶段 3：把独立比例实际作用到页面

- 目标：让设置不仅能保存，还要真正影响目标页面范围。
- 关键实现：
  - `ui-scale.js` 扩展为“root 默认比例 + 容器相对 zoom 覆盖”模型。
  - 默认比例继续作用于 `document.documentElement` 的 `font-size`。
  - 顶层 6 页的独立覆盖通过以下容器应用：
    - `#tab-config`
    - `#tab-datasets`
    - `#tab-training`
    - `#tab-weight-analysis`
    - `#tab-settings`
    - `#tab-environment`
  - 历史详情的独立覆盖只作用于 `#history-detail-content`，不影响标题栏、工具栏和页签条。
  - `history-detail/dialog.js` 在切换历史详情页签时即时重新应用对应比例。
- 工程判断：
  - 没有采用对子容器改 `font-size` 的方式，因为现有前端大量使用 `rem`，子容器字体基准不能稳定覆盖整个页面。
  - 局部覆盖采用 `zoom = override / base`，只做相对缩放，避免与默认比例相互冲掉。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py`
  - 结果：`51 passed in 15.78s`

## 总体验证

- 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_ui_scale_settings.py tests/test_preview_service.py tests/test_training_frontend_state.py`
- 结果：`76 passed in 20.75s`
- 已执行：`git diff --check -- web/services/settings_service.py tests/test_ui_scale_settings.py tests/test_preview_service.py tests/test_training_frontend_state.py web/static/js/config/catalog/defaults.js web/static/index.html web/static/css/30-preview-settings-dialogs.css web/static/js/features/anima-app/chunks/02-ensure-history-detail-feature.js web/static/js/features/anima-app/chunks/26-load-global-settings.js web/static/js/features/anima-app/chunks/36-setup-event-listeners.js web/static/js/features/app-shell/ui-scale.js web/static/js/features/history-detail/dialog.js docs/findings/ui_scale_independent_settings.md`
- 结果：通过，无空白符和 patch 结构问题。
