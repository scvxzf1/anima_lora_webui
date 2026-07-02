# WebUI 历史详情性能推进记录

## 背景

历史任务详情、样张与权重、日志、配置等视图打开时存在可感知卡顿。当前推进按阶段落地，
每个阶段都记录实现范围和真实验证命令。

## 阶段 1：预览 API 轻量读取历史任务

- 目标：预览相关 API 不再为了读取 `sample_dir`、`output_dir` 等目录元信息而调用完整
  `get_history_task()`。
- 改动：
  - 新增 `TrainingService.get_history_task_summary()`，只读取 `meta.json` 并返回 task summary。
  - 单任务预览选择改用 lightweight summary。
  - 配置组预览直接复用 `list_history_tasks()` 返回的 summary，避免逐条展开完整详情。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_preview_service.py`
  - 结果：`21 passed in 6.32s`

## 阶段 2：日志渲染分批化

- 目标：主日志和历史详情日志不再一次性创建几千个 DOM 节点，降低打开历史详情、
  切换到“日志”页签或回放训练日志时的主线程阻塞。
- 改动：
  - 主日志 `renderLogOutputLines()` 改为按批次 append，并用 `logRenderToken` 取消旧批次。
  - 主日志新增 `logOutputLines` 作为逻辑行缓存，避免异步渲染期间从半成品 DOM 读取日志。
  - 历史详情日志 `renderConsole()` 改为按批次 append，高亮搜索结果完成后再启用匹配导航。
  - 清空日志和返回实时训练时统一调用 `resetLogOutputLines()`，同步清理 DOM、缓存和旧批次。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py`
  - 结果：`50 passed in 14.28s`

## 阶段 3：详情页签内容缓存

- 目标：历史详情弹窗内切换“概览 / 训练分析 / 样张与权重 / 日志 / 配置与文件”时，
  避免已构建过的页签内容被反复销毁和重建。
- 改动：
  - `dialog.js` 内新增当前 payload 级别的 `contentCache`，按页签缓存已生成的 DOM 节点。
  - 页签点击改走 `selectHistoryDetailTab()`，只更新页签状态和内容区，不重绘标题、元信息和操作栏。
  - 同一页签内部控件触发的 `renderHistoryDetailContent()` 默认仍强制刷新，避免曲线筛选、
    续训权重状态等动态内容变旧。
  - 切换任务、关闭弹窗或清空历史详情状态时清理缓存，并保留预览 workspace 的 restore 流程。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py`
  - 结果：`50 passed in 16.13s`

## 阶段 4：历史预览图片减少 eager 加载

- 目标：打开“样张与权重”时避免历史图片缩略图一次 eager 过多，减少图片解码、
  布局和网络请求对主线程的挤压。
- 改动：
  - 新增 `HISTORY_PREVIEW_EAGER_IMAGE_LIMIT = 16`。
  - 历史任务或历史配置组选中时，仅前 16 张预览图使用 `loading="eager"`，其余保持 lazy。
- 验证：
  - 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py`
  - 结果：`50 passed in 16.25s`

## 总体验证

- 已执行：`timeout 60 .venv/bin/python -m pytest tests/test_preview_service.py tests/test_training_frontend_state.py`
- 结果：`71 passed in 21.38s`
- 已执行：`git diff --check` 覆盖本次修改的已跟踪文件。
- 已执行：`git diff --check --no-index -- /dev/null docs/findings/training_history_detail_performance.md`
