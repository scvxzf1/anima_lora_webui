# 上帝文件治理 GOAL 推进报告

日期：2026-06-07
依据：`docs/findings/god_files_refactor_goals_20260607.md`

## 执行结论

本轮完成了 GOAL-00，并推进完成 GOAL-01 中最稳的一批低风险纯函数拆分。

- GOAL-00：已建立基线说明和前端防回归护栏。
- GOAL-01：已拆出 GPU 工具、progress / metric 解析工具、config path 工具。
- 用户可见行为：保持不变。
- 旧导入路径：保留兼容 facade。
- 未触碰内容：训练历史、队列、输出、模型、数据集内容。

## 本轮改动

### GOAL-00 基线和护栏

- 在目标文档中记录四个重点文件基线行数和现有测试依赖。
- 在 `tests/test_training_frontend_state.py` 增加静态护栏：
  - `legacy-app.js` 必须声明过渡层职责。
  - `legacy-app.js` 不允许本轮后净增行数。
  - 新 feature 目录必须从生产入口模块图可达。
  - `app.js` 继续保持 bootstrap，不承载 `fetch`、DOM 查询或事件绑定业务。

### GOAL-01 后端纯函数拆分

新增模块：

- `web/services/training/gpu.py`
  - GPU 白名单 normalize。
  - `CUDA_VISIBLE_DEVICES` 环境注入。
  - `nvidia-smi` 统计和 GPU 列表解析。

- `web/services/training/progress_parser.py`
  - step rate 中位数计算。
  - progress JSONL event 转 metric。
  - tqdm / log metric 文本解析。
  - metric 去重 key、display step 分配等纯逻辑。

- `web/services/config/paths.py`
  - config 相对路径 normalize。
  - configs 边界内 safe resolve。
  - config 子目录 safe resolve。
  - 项目路径 resolve 和 display path。

兼容策略：

- `web/services/training_service.py` 保留原 `_get_gpu_stats`、`_normalize_gpu_whitelist`、`_metric_from_progress_jsonl_event` 等旧函数名，内部转调新模块。
- `web/services/config_service.py` 保留 `_safe_resolve`、`_safe_config_subdir`、`_resolve_project_path`、`_display_path` 等旧函数名，内部转调新模块。

## 行数变化

- `web/static/js/features/legacy-app.js`：17882 行，保持不变。
- `web/static/style.css`：15528 行，保持不变。
- `web/services/training_service.py`：5635 行 -> 5400 行。
- `web/services/config_service.py`：4242 行 -> 4223 行。

## 验证结果

已通过：

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py
timeout 60 .venv/bin/python -m pytest tests/test_training_gpu_selection.py
timeout 60 .venv/bin/python -m pytest tests/test_web_config_service.py
timeout 60 .venv/bin/python -m pytest tests/test_training_queue.py
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py
python -m py_compile web/services/training_service.py web/services/config_service.py web/services/training/gpu.py web/services/training/progress_parser.py web/services/config/paths.py
git diff --check -- web/services/training_service.py web/services/config_service.py web/services/training web/services/config tests/test_training_frontend_state.py tests/test_training_gpu_selection.py tests/test_training_resume.py tests/test_web_config_service.py docs/findings/god_files_refactor_goals_20260607.md
```

说明：裸 `python` 环境缺少 `toml` 依赖，后端 pytest 使用项目 `.venv/bin/python` 复跑并通过。

## 工作区注意事项

开始本轮前，工作区已有大量运行期配置和训练历史改动，包括：

- `configs/web-training-queue/`
- `configs/web-training-history/`
- `configs/datasets/`
- `configs/imported/`
- `configs/sample-prompts/imported/`

本轮没有回滚、删除或修改这些运行期数据。

## 后续建议

下一步建议按原文档顺序继续：

1. GOAL-02：拆 `config_service.py` 的 methods / sample prompts / output runs / raw files。
2. GOAL-03：拆 `training_service.py` 的 queue store、history store、runtime paths。
3. GOAL-04：拆 `legacy-app.js` 中主题、GPU picker、standalone warning 等轻量 feature。

不建议直接跳 GOAL-05/06/07，因为配置表单、CSS 和训练历史 UI 都依赖前面 facade 边界稳定。
