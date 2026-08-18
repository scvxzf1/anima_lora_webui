# Dragon Next 前端审计与修复（2026-08-19）

状态：已完成一轮全功能审计与修复
日期：2026-08-19
范围：`web/frontend-next`（`/next` SPA）全部 feature，含数据集、训练配置、队列、当前监控、历史。

## 1. 审计方法

- 静态审查：逐 feature 对照后端真实契约（`curl` 实际返回），重点排查 API shape、hooks、dirty/guard、query key。
- 真实浏览器：Playwright + headless Chrome 加载 `/next` 全部路由，收集 pageerror 与 console.error。
- 隔离环境：用 `ANIMA_CONFIGS_ROOT=/tmp/dragon-next-audit/configs` 启动第二个 WebUI（端口 20299），在该隔离配置根内实测数据集分组 CRUD、新建/保存/重命名/删除预设、导入、应用到训练配置、队列暂停与策略保存，避免触碰用户运行数据。

## 2. 已修复问题

| # | 文件 | 问题 | 修复 |
| --- | --- | --- | --- |
| 1 | `src/api/trainingContext.ts` | `/api/presets` 实际返回 `{ok, items}`，前端按字符串数组读取，导致 `/next/training` 与 `/next/datasets` 在真实数据下白屏崩溃 | `fetchTrainingPresets` 兼容 envelope 与数组，并更新测试 |
| 2 | `src/features/live-monitor/LiveMonitorPage.tsx` | WS `progress`/`log`/`system` 消息按错误形状解析（`message.progress`、`message.record`），真实广播为平铺结构，WS 增量完全失效 | 按真实广播形状解析：progress 取 `message.progress \|\| message`，log 直接使用消息本身，system 写入 `latest_system` |
| 3 | `src/features/dataset-editor/useDatasetPresetEditor.ts` | 保存/另存/重命名后 `selectedFile` 指向新文件，但旧 `presetPaths` 未及时更新，effect 将选择跳回第一个预设 | effect 只处理“无选择”和“预设列表为空”，不再因 `presetPaths` 缺失回退 |
| 4 | `src/features/training-history/HistoryDetailPage.tsx` | 详情页状态显示英文 `running/error`，与列表页中文状态不一致 | 增加状态中文映射 |
| 5 | `src/app/useWebSocket.ts` | 连接/关闭/错误回调在卸载后仍可能 setState | 回调入口统一检查 `disposed` |
| 6 | `web/frontend-next/index.html` | 无 favicon 链接，控制台持续报 `/favicon.ico` 404 | 引用 `/static/favicon.svg` |
| 7 | `src/features/training-queue/QueuePage.tsx` | 缺少暂停/继续队列与手动刷新入口 | 头部增加暂停/继续、刷新按钮 |

## 3. 隔离环境浏览器验收

- `/next/datasets`：分组创建、重命名、删除通过；新建草稿并保存、重命名、删除预设通过；导入预设通过；应用到当前训练配置通过。
- `/next/queue`：暂停队列、保存失败与重试策略通过。
- `/next/training`：真实字段渲染、编辑字段后“预览变更”通过。
- `/next/history`：搜索、状态筛选、批量选择、详情页深链刷新通过。
- `/next/monitor`：真实运行日志渲染，WS 增量与轮询共存，无控制台错误。
- 所有路由 `pageerror`/`console.error` 均为 0（favicon 修复后）。

## 4. 门禁

```text
pnpm --dir web/frontend-next typecheck  通过
pnpm --dir web/frontend-next test       68 passed / 13 files
pnpm --dir web/frontend-next build      通过
```

## 5. 仍未纳入本轮的范围（后续阶段）

- 历史集合/配置组工作台、样张/权重 artifacts、续训与排队续训入口。
- 当前监控 Loss/系统指标图表、日志搜索/复制/下载。
- 队列运行目录清理删除与续训入队。
- 以上对应功能对照矩阵仍为“迁移中”，补齐后升为“已验证”。
