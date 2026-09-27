# Dragon Next S3 队列、监控、历史与续训控制面验证报告

状态：**PASS（仅 S3；非发布验收）**
日期：2026-09-25
范围：`/next/` React 工作台的队列、当前监控、训练历史和续训控制面。Classic/旧 Dragon、真实 GPU 启动/停止/续训、外部 provider 和用户数据写入不在本轮范围。

## 基线与安全边界

- 分支/HEAD：`dev` / `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`。
- 工作树：验证开始前已存在大量未提交修改；`git status --short` 记录为非 clean（本轮不 reset、不清理、不覆盖）。新增 S3 证据只涉及本报告、索引、总审计 S3 段落和两份测试文件。
- 静态包/服务：未运行发布构建；Playwright 使用当前工作树 Vite 隔离服务 `http://127.0.0.1:20627/next/`，没有替换 `web/static/dragon-next/`。
- E2E fixture：`mockWorkspace` 的内存 HTTP/WS route；非 GET 请求只由测试 route 接收，不写仓库配置、队列、历史或输出根。
- Python fixture：`tmp_path` 下的临时 queue/history/output 根；假进程只验证服务控制分支，不启动 launcher、GPU 或训练子进程。
- 浏览器：Playwright Chrome headless。既有矩阵覆盖 `1440x900`、`1280x720`、`768x1024`、`390x844` 的 dark/light；S3 新增用例随同集合执行。截图/trace 输出在 `/tmp/dragon-next-s3-e2e-final`，无失败用例因此没有最终失败 trace。

## 实施内容

| 证据 | 内容 |
| --- | --- |
| `web/frontend-next/e2e/s3-queue-monitor-history.spec.ts` | 队列五状态矩阵、确认范围与 `expected_revision`、刷新后的持久快照；历史产物 `available/missing/blocked/unreadable`；检查点总目标到追加步数和排队续训 payload。 |
| `tests/test_dragon_next_s3_control_plane.py` | 临时 queue JSON 重启恢复、陈旧 `expected_task_id` 停止拒绝、临时 history archive/unarchive 重启恢复。 |
| 本次 S3 审计修复 | 队列持久化改为通过 `set_queue_settings()` 控制命令写盘并核验重启后的 revision/旧 revision 拒绝；停止测试加入 active fake process、stale ID 无副作用与匹配 ID 清理；产物测试核验下载 href/`download` 属性及不可用状态不生成链接；补齐 Findings 索引并关闭 V-01 历史表述矛盾。 |

## 执行命令与结果

### Mock E2E

```bash
DRAGON_E2E_PORT=20627 pnpm --dir web/frontend-next exec playwright test \
  queue-monitor-mutations.spec.ts queue-scope.spec.ts state-feedback.spec.ts \
  history-return.spec.ts history-overview-assets.spec.ts history-logs.spec.ts \
  overview-states.spec.ts s3-queue-monitor-history.spec.ts \
  --output=/tmp/dragon-next-s3-e2e-final
```

结果：**50 passed，0 failed，约 2.4 分钟**。

- 队列：等待/运行/异常/完成/取消状态、批量范围确认、revision 竞争、无 revision fail-closed、慢请求、缓存快照和结果未知均通过。
- 监控：HTTP status/metrics/logs 局部 5xx、旧 task WS 事件过滤、停止请求 `task_id` 绑定、停止期间锁定、任务切换、断线重连和结果未知均通过。
- 历史：搜索游标、深链返回位置、删除锚点回退、恢复失败可重试、产物分页/缺失提示和任务状态矩阵均通过。
- 续训：冲突后必须重新检查并重新确认；S3 新用例验证 step `40`、总目标 `120` 发送为 `duration_overrides.max_train_steps = 80`，并且排队目标与任务 ID 正确。

### 隔离服务/假 launcher

```bash
.venv/bin/python -m pytest -q \
  tests/test_dragon_next_s3_control_plane.py \
  tests/test_training_queue_resume.py tests/test_training_resume_actions.py \
  tests/test_training_resume_options.py tests/test_training_queue.py \
  tests/test_training_task_lifecycle.py tests/test_history_cursor_contract.py \
  tests/test_training_history_artifacts.py tests/test_history_artifact_manifest.py \
  tests/test_training_history_delete.py
```

结果：**120 passed in 10.94s**。

- queue 状态写入 `queue.json` 后由新 `TrainingService` 实例恢复，暂停标志、任务顺序和五类状态计数一致。
- `stop(expected_task_id=...)` 对陈旧任务 ID fail-closed；匹配当前任务才进入停止分支。
- 历史归档/取消归档在临时 history 根落盘，重启服务后状态保持；删除、恢复、游标非法和路径越界均由同一隔离测试集覆盖。
- 检查点完整性要求 `train_state.json`、model、optimizer、scheduler（按需）和 random state；缺失状态不会调用 launcher。schedule-free 的 scheduler 例外由现有契约测试单独覆盖。

### 前端工程门禁

```bash
pnpm --dir web/frontend-next typecheck
pnpm --dir web/frontend-next test
```

结果：typecheck **PASS**；Vitest **49 files / 218 tests passed**。

### 本次修复回归

```bash
.venv/bin/python -m pytest -q tests/test_dragon_next_s3_control_plane.py
pnpm --dir web/frontend-next exec playwright test s3-queue-monitor-history.spec.ts
.venv/bin/python -m pytest -q tests/test_documentation_integrity.py
```

结果：S3 控制面 **3 passed**；S3 Playwright **2 passed**；文档完整性 **8 passed**。随后按本报告隔离服务/假 launcher 集合重跑，结果仍为 **120 passed in 11.31s**。

## S3 验证矩阵

| 控制面 | 状态 | 关键证据 |
| --- | --- | --- |
| 队列状态与批量范围 | **PASS** | 五状态矩阵；确认弹窗范围；`expected_revision`；409/未知结果后刷新恢复；按钮锁定。 |
| 当前监控 | **PASS** | 指标/日志按确认的 `task_id` 查询；旧 WS 事件丢弃；局部失败保留旧数据并可重试；断线后 HTTP 快照收敛。 |
| 历史分页与返回 | **PASS** | 游标分页、深链筛选/滚动恢复、缺失锚点显式回退、恢复失败可重试。 |
| 历史产物 | **PASS** | `available/missing/blocked/unreadable` 均有独立可见状态；路径白名单和越界拒绝通过。 |
| 检查点与续训 | **PASS** | 完整性门禁；总目标减检查点 step 的追加步数换算；409 后重新读取/确认。 |
| 临时根持久化 | **PASS** | queue/history 重启恢复；只使用 pytest 临时目录。 |
| 真实服务写入、GPU 启动/停止/续训 | **NOT RUN** | S3 明确排除；需用户另行授权，不以 mock 结果替代。 |

## 问题与未执行项

- P0：`N/A`。
- P1：`N/A`。
- P2：`N/A`。
- G1：`N/A`（本轮 S3 证据门禁已完成；S0 的生产只读脚本历史阻塞不在本轮重判）。
- 仓库级 `tests/test_documentation_integrity.py` 已恢复为 **8 passed**；`adaptive_runtime_20260921.md`、`adaptive_training_20260922.md` 和 `dragon_next_s2_isolation_20260925.md` 已补入 `docs/findings/README.md` 索引。索引修复不改变这些文档自身内容或阶段状态。
- 过程中曾出现一次测试选择器与统计按钮可访问名称不一致，已仅修正新增测试选择器；未发现产品行为缺陷。
- S1 的完整人工/读屏器矩阵、S2 配置/数据集/模型/设置写入、S4 图片/蒙版/打标和 S5 发布候选仍保持各自原状态，本报告不替代这些阶段。

## 回退与签收

本轮没有修改运行时数据、模型、真实队列、历史或输出。若需回退，只需移除本报告、索引/总审计中的 S3 记录以及两份新增 S3 测试文件，不涉及数据迁移。

签收人：Codex 当前任务（工程证据）；用户发布签收：未进行。
