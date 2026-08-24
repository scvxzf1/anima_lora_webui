# Dragon 响应式硬编码审计（2026-08-24）

状态：已收口

## 审计范围

本轮检查 `web/static/css/dragon/` 和 `web/static/js/dragon-ui/` 中会影响布局、视口判断和页面初始化的固定值。图标尺寸、可点击高度、边框和组件内间距保留固定值；这些值是视觉和可用性契约，不是需要消除的布局耦合。

## 已处理

- 新增 `responsive.js`，集中命名 JS 使用的导航和训练预设侧栏断点，避免特性模块重复写裸数字。CSS media query 仍留在对应组件附近，便于局部维护。
- `IntersectionObserver` 不可用时直接显示所有滚动揭示内容，不再因可选动效导致 Dragon 初始化失败。
- `requestAnimationFrame` 不可用时使用短定时器驱动视差更新，保持基本功能。
- 页面根容器、训练对话框和数据集侧栏采用 `vh` 回退 + `dvh` 增强的双声明，兼顾旧引擎和移动浏览器动态工具栏。
- 训练对话框宽度由 `100vw` 改为包含块百分比，避免桌面端滚动条宽度被重复计入。

## 保留边界

- CSS 组件断点不能通过普通 custom property 与 JS 共享；为保持当前浏览器兼容面，未引入支持度仍不稳定的 custom media 构建链。
- 对话框、导航和编辑器内部仍有明确的最小尺寸，用于防止操作区压缩到不可用。
- 本轮不改写历史页的单行压缩 CSS，也不借机重排整个样式树。

## 验证入口

```bash
timeout 60 .venv/bin/python -m pytest -q \
  tests/test_dragon_motion_settings.py \
  tests/test_dragon_monitor_system_frontend.py \
  tests/test_dragon_model_quick_picker_frontend.py \
  tests/test_dragon_dataset_editor_frontend.py
```

同时对 Dragon JS 树执行 `node --check`，并在宽屏、手机窄屏和矮视口下检查横向溢出、导航切换和对话框可访问性。
