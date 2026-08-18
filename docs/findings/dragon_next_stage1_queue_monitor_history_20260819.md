# Dragon Next 新前端首版迁移：队列 / 当前监控 / 历史

状态：首版已实现并验证
日期：2026-08-19
范围：`web/frontend-next`（`/next` SPA），不修改 Classic 与 vanilla Dragon。

## 1. 本阶段完成

| 页面 | 路由 | 主要能力 |
| --- | --- | --- |
| 训练队列 | `/next/queue` | 队列统计与状态筛选、暂停/继续队列、失败与重试策略表单、单条置顶/上移/下移/置底、重试、取消/停止、移出列表、批量取消等待/清理完成/清理取消、中止后续队列、强制中止 |
| 当前监控 | `/next/monitor` | 训练状态、进度条、Loss/学习率/速度/显存/GPU 温度/利用率、实时日志、GPU 设备列表、停止训练；REST 轮询 + `/ws/training` 增量日志/进度 |
| 历史任务 | `/next/history` | 总量/训练/预处理/异常/归档统计、搜索、状态与归档筛选、批量归档/取消归档/彻底删除、详情页（任务信息、日志、配置快照） |
| 历史详情 | `/next/history/:taskId` | 任务信息、最终 Loss/步数/学习率/曲线点数、日志查看、TOML 配置快照 |

共享改造：

- 新增 `app/Topbar.tsx`，统一五入口导航（数据集蓝图 / 训练配置 / 训练队列 / 当前监控 / 历史任务）。
- 新增 `app/useWebSocket.ts`，通用 WS 连接、指数退避重连、卸载清理。
- `DatasetWorkspace` 与 `TrainingWorkspace` 改用共享 `Topbar`，不再各自复制导航。

## 2. 与 Classic 的差距（后续阶段）

- 队列：暂未做历史队列项归档/运行目录清理删除，也未做失败项自动重试后的后台调度展示细分。
- 当前监控：暂未做日志搜索/复制/下载、Loss 曲线 SVG、指标历史图表；WS 只用于进度与日志增量，断线后依赖轮询兜底。
- 历史：暂未做集合/配置组工作台、时间线、样张与权重 artifacts、续训/排队续训入口、任务详情 tabs 的完整切换。
- 所有危险操作均有二次确认，但删除历史仅删除记录，不删除运行目录和权重。

## 3. 验证

```text
pnpm --dir web/frontend-next test
Test Files  13 passed (13)
Tests       68 passed (68)

pnpm --dir web/frontend-next typecheck
通过

pnpm --dir web/frontend-next build
通过，产物输出到 web/static/dragon-next（该目录被 .gitignore 忽略，不提交）
```

新增测试文件：

- `src/features/training-queue/api.test.ts`
- `src/features/training-queue/QueueItemCard.test.tsx`
- `src/features/live-monitor/api.test.ts`
- `src/features/training-history/api.test.ts`

## 4. 关键实现位置

- `src/app/router.tsx` — 新增 `/queue`、`/monitor`、`/history`、`/history/:taskId` 四个 lazy route。
- `src/features/training-queue/` — QueuePage / QueueItemCard / QueuePolicyForm / api / types / css。
- `src/features/live-monitor/` — LiveMonitorPage / api / css。
- `src/features/training-history/` — HistoryPage / HistoryDetailPage / api / css。

## 5. 下一步

1. 历史集合/配置组工作台与详情 tabs 补全。
2. 当前监控 Loss/系统指标图表与日志搜索/下载。
3. 队列与历史中的续训/排队续训入口，接 `resume-options` 与共享训练上下文。
4. 完成 Classic 对照矩阵中对应项 `未核验 → 已验证` 的门禁升级。
