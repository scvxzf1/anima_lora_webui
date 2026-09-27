# Dragon Next S1 壳与导航审计进度（2026-09-25）

状态：NOT RUN（S1 整体退出条件尚未全部核对；已完成部分壳层、焦点和 mock 回归）；不代表发布验收

## 当前快照（2026-09-27，r115 全量 / r111-r114 增量复核）

- S1 仍为 **NOT RUN / 未签收**。当前树最新完整 mock API/WS Playwright r115：**303 项，301 passed、2 skipped、0 failed**；2 个 skip 是 `dataset-drag.spec.ts` 的折叠分组 hover/drop 场景。r115 使用隔离 Vite 端口 `5199`、不可达 API target `127.0.0.1:29999`、2 workers；不连接真实 API/WS。
- r94 新增训练配置专属 409 mock E2E：配置保存首次返回 409 时草稿保留、保存按钮锁定且只发出一次 PATCH；用户确认“重新加载配置”后采用服务器 `server-v2` revision，随后再次编辑才以该 revision 保存。`training-draft-flow.spec.ts` 完整定向 **6/6 passed**，产物 `/tmp/dragon-next-e2e-s1-training-config-409-r94/`。
- r95 新增图片工作台读取错误 mock E2E：持续 503 时显示预览错误且不伪造图片卡片，默认 query 重试窗口结束后只由显式“重试”恢复；`mask-workspace.spec.ts` 完整定向 **14/14 passed**，产物 `/tmp/dragon-next-e2e-s1-dataset-image-error-r95/`。只读图片 GET 的 `offline_unknown` 不属于当前产品契约，矩阵已登记为 `N/A`。
- r96 新增 Captioning confirmed-empty mock E2E：打标任务列表返回空数组时显示“暂无打标任务”，图片来源和“新任务”入口仍可用；`captioning-prompts-states.spec.ts` 完整定向 **3/3 passed**，产物 `/tmp/dragon-next-e2e-s1-captioning-empty-r96/`，无 provider/写请求。
- r97 新增设置/模型状态反馈 mock E2E：设置读取挂起时表单保持不可用，保存 pending 锁定表单；模型读取挂起时编辑器不可用，成功读取暴露已同步默认模型。`state-feedback.spec.ts` 完整定向 **23/23 passed**，产物 `/tmp/dragon-next-e2e-s1-settings-model-loading-r97/`。
- r98 新增三组初始读取 gate mock E2E：数据集库预设 GET 挂起时保持 `aria-busy` 与“正在读取预设库”；蒙版列表→图片两阶段读取期间画布与保存/应用命令不可用；监控状态释放后，指标/日志/GPU 仍分别显示读取中且没有陈旧确认内容。三个 spec 合计 **33/33 passed**，产物 `/tmp/dragon-next-e2e-s1-loading-r98/`，无 fixture writes/未处理请求。
- r99 新增训练 loading/empty/success 与历史 success mock E2E：训练上下文 gate 保持 selector 与命令禁用，空配置库不允许执行，正常配置暴露完整命令面；历史完成态在同一 task identity 下串联摘要、产物分页和日志。`training-draft-flow.spec.ts` 与 `history-overview-assets.spec.ts` 合计 **22/22 passed**，产物 `/tmp/dragon-next-e2e-s1-training-history-r99/`。
- r100-r102 补齐 Captioning loading/success：profiles/jobs/prompts 挂起期间来源页保持不可提交，释放后恢复；已完成候选选中后 mock commit 反馈“写入 1 / 冲突 0”。r100 首次全量定向发现桌面隐藏任务切换按钮的 locator 假设，未暴露产品失败；修正后 loading **1/1 passed**（r102，产物 `/tmp/dragon-next-e2e-s1-captioning-loading-r102/`），success 在 r100 **1/1 passed**。未调用 provider 或真实 TXT。
- r103 重跑 Captioning 完整定向集合 **26/26 passed**，确认 r102 locator 修复与 success commit 在完整上下文下稳定；产物 `/tmp/dragon-next-e2e-s1-captioning-r103/`。
- r104 收敛图片工作台 busy 语义：大图 viewer 在真实 `<img>` `load` 事件前暴露 `aria-busy="true"` 与“正在读取图片”，失败先结束 pending 再显示可重试错误；新增 `DatasetImageViewer.test.tsx` 两项组件级 mock 生命周期核验，图片/蒙版 E2E **15/15 passed**，六组重点 S1 spec 合计 **76/76 passed**，未触碰图片服务或写入。
- r105 扩展 history success mock E2E：完成态详情现在串联产物/日志身份、可恢复 checkpoint 读取、确认后的 resume POST body 与“查看监控”结果；不连接真实 checkpoint、队列或训练服务。
- r106 收敛训练设备读取语义：设备快捷选择区域现在暴露 GPU 首读/刷新期间的 `aria-busy`；新增挂起 GPU GET 的 mock E2E，确认“正在读取 GPU”期间执行命令仍禁用，释放后恢复设备选择。
- r107 修复 Captioning 读取错误的恢复语义：profiles/jobs query 停止失败后的自动重试/轮询，并通过 QueryFeedback 提供显式“重试接入预设/重试打标任务”；详情页已有的“重试打标任务”也补上浏览器证据。新增 provider library、job library、job detail 三个失败恢复场景，Captioning spec **7/7 passed**。
- r107 补充 history 完成态的组合完整性：同一完成任务同时包含可用配置、不可读日志、缺失运行配置，以及样张/权重目录不存在；摘要显示目录缺失，结果文件不为不可读日志生成下载链接。`history-overview-assets.spec.ts` **14/14 passed**。
- r107 为 dataset preview API 增加非默认 `limit` 的 URL 契约回归；不新增 `source=training` 或 offset 产品行为。API 单测 **10/10 passed**；修正既有 `DatasetImageViewer` 测试 fixture 后 TypeScript 通过。
- 最近一次前端验证：TypeScript 通过；串行 Vitest **61 文件 / 256 用例通过**；`rtk git diff --check` 和矩阵校验通过。r111-r112 新增增量 spec 合计 **4/4 passed**，r114 对 r113 两个并发波动文件串行复核 **9/9 passed**；不证明真实 API/WS、launcher、provider 或持久化。
- r108 完整当前树回归 **292 项：290 passed、2 skipped、0 failed**，耗时约 10.1 分钟，产物 `/tmp/dragon-next-e2e-s1-r108/`；全量包含 r107 的 Captioning/history 增量。完整 mock 通过不外推为真实服务、launcher、provider、持久态或人工读屏/视觉通过。
- r109 新增 Captioning profile 状态回归 **2/2 passed**，覆盖服务端密钥不回显/不重提交，以及保存 503 后保留名称和密钥草稿、显式重试恢复；产物 `/tmp/dragon-next-e2e-s1-caption-profile-r109/`。仍只使用 route mocks，不连接 provider 或写入真实配置。
- r109 补充蒙版非零分页回归：完整 `mask-workspace.spec.ts` **16/16 passed**，验证 `offset=48`、第二页图片身份和返回第一页缓存；产物 `/tmp/dragon-next-e2e-s1-mask-pagination-r109-r3/`。未触碰真实图片、蒙版或用户目录。
- r110 在当前工作树完成完整 mock API/WS Playwright 回归 **295 项：293 passed、2 skipped、0 failed**，耗时约 10.2 分钟，产物 `/tmp/dragon-next-e2e-s1-r110/`；全量包含 r109 的 Captioning profile 与蒙版分页增量。完整 mock 通过不外推为真实服务、launcher、provider、持久态或人工读屏/视觉通过。
- r111 修复 Captioning 初始建 job 的协议边界：缺少 `job.id` 的 202 响应现在 fail-closed 显示错误，不导航到不存在的任务；新增 503 保留选择后显式重试成功、请求 payload 与缺少任务 ID 两项 mock E2E，**2/2 passed**。产物 `/tmp/dragon-next-e2e-s1-caption-create-r111-r2/`。
- r112 新增历史续训边界 **2/2 passed**：空 checkpoint 保持确认与 POST 禁用；响应丢失显示“核对历史任务/核对队列”链接并禁止盲重试。产物 `/tmp/dragon-next-e2e-s1-history-resume-edge-r112/`。
- r113 首次并发全量为 **300 项，296 passed、2 skipped、2 failed**；失败仅为既有 `dataset-scroll`、`history-grouping` fixture 等待超时。r114 将两份完整定向 spec 串行重跑为 **9/9 passed**，不将 r113 作为最终快照。
- r115 在空闲环境完成当前树完整 mock API/WS Playwright 回归 **303 项：301 passed、2 skipped、0 failed**，耗时约 12.5 分钟，产物 `/tmp/dragon-next-e2e-s1-r115/`；全量包含 r111-r112 新增证据及当前树其他增量。完整 mock 通过不外推为真实服务、launcher、provider、持久态或人工读屏/视觉通过。
- 已登记机器可核验矩阵：[dragon_next_s1_state_matrix_r93.json](dragon_next_s1_state_matrix_r93.json)（文件名沿用，内部 `snapshot=r115`），共 70 个 workspace/state 单元格；既有状态证据仍为 65 个 `PASS_EVIDENCE`，5 个当前契约不适用单元为 `N/A`，r109-r115 的 profile 安全、蒙版分页、建 job 与续训边界作为附加契约证据记录，不新增状态行。当前为 65 个 `PASS_EVIDENCE`、0 个 `MANUAL_REQUIRED`、5 个 `N/A`；`validate-s1-state-matrix.mjs` 校验文件、测试标题和必填边界通过后，矩阵 `overall_state` 仍为 `NOT_RUN`，它是证据登记，不是 S1 签收。

- 当前 query 根切换覆盖已扩展到 training raw/estimate/picker-preview/preflight、history list/detail/artifact/images/weights/resume-options/collections、日志 metadata/offset page/search，以及 captioning 来源 dataset library/preset/images 的 offset=0/60。r91 另核验 dataset preview/mask 的第二子集 `dataset_index=1`、preview `source=source&limit=120`、mask `offset=0` 与 image 身份在根切换时不串用；Captioning 本地资源/词典读取连续失败后保持错误，仅显式刷新恢复。r80 补充数据集导入 409/响应丢失语义与 pending 锁；r84 补充 history log search；r85 补充蒙版缩放状态的读屏反馈。captioning 非 dataset 来源的存储根不同，细节见下文 Query Key 对账边界。
- 矩阵的未运行/人工格子、实体读屏器/live-region 顺序、剩余视口/主题/真实缩放人工检查、真实 launcher/服务和隔离持久态仍未完成。工作树在 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`、ahead 13 且包含大量既存修改；本轮不清理、不暂存、不提交。

## 已执行

- 分支 `dev`，HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`；工作树仍有多处既存修改，另有本轮 S0/S1 记录与前端 E2E 改动。没有连接真实服务或执行真实写入。
- 历史完整 Playwright mock API/WS suite r17：**184 项中 182 passed、2 skipped、0 failed**，单 worker 耗时 13.4 分钟。使用隔离 Vite 端口 `21429`、不可达 API target `127.0.0.1:29999`；报告和 trace 在 `/tmp/dragon-next-e2e-s1-20260925-r17/`。2 个 skip 是 `dataset-drag.spec.ts` 中已有的折叠分组拖动场景；计数包括 S2/S3 mock spec，不是真实 API/WS 集成证据。
- 其中旧 `dataset-draft.spec.ts` 两个视口用例原先查找不存在的旧版“打标工作台”链接。当前实现是在 dirty 状态禁用“图片工作台”按钮，因此将 E2E 改为断言按钮禁用、提示先保存、草稿和 URL 保持不变。1440/390 两个变体定向通过。
- 新增 `app-shell.spec.ts`，覆盖移动主导航展开/收起、当前导航标识、浏览器 back/forward、主题切换跨路由及 reload 持久化、系统减少动态效果初始/运行时切换、skip-link 聚焦工作区、未知路由恢复，以及训练/数据集 dirty guard 的 skip-link 历史回归；最新定向 6/6 通过。仅在 Playwright 临时浏览器上下文写主题 localStorage，fixture 记录的 API mutation 为 0。
- 曾用隔离 fixture 确认 skip-link 的 P1：原生 `href="#workspace"` 产生的非 Router history entry 让脏训练页第一次 Back 直接回到此前 `/queue`，未出现确认并离开当前草稿；同时打印 React Router POP blocker warning。已改为 Router 管理的 hash 链接，目标 `#workspace` 可程序化聚焦；两种 dirty guard 现在只放行 path/search 不变的 hash-only 导航。回归确认 hash Back 留在当前页、下一次离页触发原守卫且草稿保留，warning 不再出现。
- 定向命令 `app-shell.spec.ts e2e/dataset-draft.spec.ts`：7 passed；包含上述训练原生确认框、数据集丢弃 dialog、1440/390 dirty 保存失败场景。`web-next-check`：TypeScript 通过，Vitest 49 文件 / 221 用例通过。
- 本轮续审新增 4 项 mock-only E2E：数据集蓝图保存返回 409 后保留修改，并要求显式重新读取以核对服务器版本；历史列表 503 后显示恢复位置错误，重试后明确进入空列表；390×844 provider 弹窗焦点进入/ESC 返回触发按钮；中断 Settings 懒加载模块后展示路由错误并可导航回训练页。四项组合定向 4/4 通过，API 写入记录为零；provider 弹窗截图在 `/tmp/dragon-next-e2e-s1-20260925-r5/`，人工检查未见遮挡。
- 后续续审又补 6 项 mock-only 状态验证：模型 revision 409 取消重载保留草稿、确认后采用服务器版本；全局设置 PUT 断网只提交 patch 并显示结果未知；队列单项移动 pending 锁定、成功刷新排序、响应丢失后手动刷新对账；监控无任务/未知状态/GPU empty、停止 pending 与成功、停止断网后按状态快照对账且不自动重试。`state-feedback.spec.ts` 15/15、`queue-monitor-mutations.spec.ts` 4/4；均断言 fixture 写入记录和未处理 API 请求为空。
- 本轮再补 3 项 mock-only 控制面证据：caption 候选保存断网后保留草稿并手动重试；历史续训 409 后必须重新检查并再次确认才可重试；训练入队 pending 禁止重复提交，断网后显示结果未知且无自动重试。组合定向 3/3 通过，未连接真实 provider/队列/训练服务。
- 图片/蒙版首轮补测新增 `mask-workspace.spec.ts` 三项 mock-only E2E：画笔产生 dirty 后切换打标页，取消仍保留蒙版与 dirty，再确认后才切换；保存首请求返回 409 后保留 dirty，手动重试成功，并断言两次都带 `If-Match: mask-rev-1`、`Content-Type: image/png` 和非空 PNG body；应用到子集首请求 pending 时锁定确认/取消，409 后对话框保留且显式重试成功。首轮定向 3/3 通过；没有蒙版 API 写入落到通用 fixture，也没有连接真实 API。
- 本轮续审再新增 3 项 mock-only 用例：蒙版 dirty 时切换图片/子集，取消后原上下文与 dirty 保留、确认后重新 GET 目标 `dataset_index`/`image` 且无 PUT；Next caption TXT commit 断网后保留选中项、pending 锁定、不自动重试，用户显式重试后成功；训练立即启动的 500/断网失败锁住当前提交，断网显示结果未知，关闭并重开弹窗后重新预检且要求重新确认。定向分别为蒙版 spec 4/4、caption 新用例 1/1、训练新用例 1/1；均使用不可达 API target 和 mock fixture。
- 本轮继续补 3 项 mock-only 覆盖：缺少 `dataset` 的图片预览/蒙版深链显示可解释空态并可返回；蒙版 apply 遇 HTTP 500 或断网时保持对话框、pending 锁定且不自动重试，显式重试成功；数据集预设库 503 显示错误，用户重试后进入明确空态。两份定向 spec 共 13/13 通过（`/tmp/dragon-next-e2e-s1-20260925-r11/`）。数据集库失败态已收敛时开发模式记录 3 个 GET；测试现在以该稳定基线断言手动重试恰好增加 1 个 GET，不把 StrictMode 重挂载和 React Query 自动重试当成产品重复请求。
- 本轮新增 `queue-policy-mutations.spec.ts` 四项 mock-only mutation 验证：保存 pending 锁住策略/队列命令并应用确认快照；HTTP 500 与断网均保留编辑、无自动重试且显式重试成功；取消“继续队列”确认时不发送请求。定向 4/4 通过（`/tmp/dragon-next-e2e-s1-queue-policy-r1/`），fixture 写入/未处理请求均为空；完整 Playwright r13 将该 spec 纳入 173 项回归。
- 本轮补齐监控三个 mock-only 缺口：`queue-monitor-mutations.spec.ts` 验证 GPU API 503 的可见错误不遮蔽任务状态/日志，用户显式重试后设备卡恢复；`useWebSocket.test.tsx` 验证连续 close 的 1/2/4 秒退避、error 等待 close 才安排重连、成功 open 后退避归零及卸载清理；`state-feedback.spec.ts` 在 1440/1280/768/390、dark/light 下校验图表、canvas、控制项均落在 viewport 内。组合定向 9/9 通过（`/tmp/dragon-next-e2e-s1-monitor-r14/`），390px dark/light 截图已抽查，未见图表、控制项、日志或设备信息遮挡/裁切。
- 本轮队列单项竞争补测新增 404 用例：确认取消等待项后模拟项目已被另一端移除，核验 DELETE 路径/body、pending 锁、旧快照保留和无自动重试；显式刷新取得空快照后清除旧动作错误并显示已核对状态。该测试促成 `QueuePage.refreshQueue()` 修复，只在显式 refetch 成功后清理动作错误；定向 1/1 通过（`/tmp/dragon-next-e2e-s1-queue-conflict-r1/`）。
- 本轮新增 caption provider ping 状态恢复 mock E2E：取消确认时零请求/零成功提示；确认后模拟真实契约的 HTTP 502/“外部 API 请求超时”，验证请求只发送 `{mode: "ping", profile_id}`、pending 时按钮禁用、错误可见、等待超过默认 mutation retry 窗口后没有自动重试，并由用户再次点击后成功。该用例未连接外部 provider，也未提交图片或写入配置。
- 模型库 mock E2E 补上默认项保护/切换、保存 pending 控件锁定与成功后 revision/默认标记同步。另修复模型保存 409 或网络结果未知后仍允许重发的问题：现在保存被锁住，提供“重新载入服务器版本”或“核对服务器版本”，用户确认后显式 GET 并采用服务器快照。断网用例模拟 PUT 已被服务端提交但响应丢失，确认草稿保留、无自动重试、核对后回到已同步状态；使用 mock API，不触及磁盘。
- 后端契约核实：未配置模型库时 GET 会从旧全局配置构造一个默认模型；已配置但 items 为空的库返回 409，PUT 也拒绝空列表。因此“空模型库可编辑态”不是有效服务端状态，本轮按错误契约处理而非虚构空态。
- 通用预览 fixture 原先缺少 `DatasetPreviewResponse.row/settings` 及完整 caption shape，与当前 TS/API 契约不符；已补齐 mock，并新增有效图片工作区预览 E2E，断言三张图片、原始路径与零 `pageerror`。完整 `mask-workspace.spec.ts` **9/9 passed**（`/tmp/dragon-next-e2e-s1-mask-regression-r2/`）。
- 后续定向：`interactions.spec.ts` **9/9 passed**（`/tmp/dragon-next-e2e-s1-caption-provider-r2/`），`state-feedback.spec.ts` **17/17 passed**（`/tmp/dragon-next-e2e-s1-model-config-r2/`）；新增 409 保存锁断言后的模型三场景 **3/3 passed**（`/tmp/dragon-next-e2e-s1-model-mutation-r3/`）；TypeScript 通过，串行 Vitest **50 文件 / 223 用例通过**，`rtk git diff --check` 通过。
- 2026-09-25 续审补测：`caption-context.spec.ts` **15/15 passed**，覆盖 rerun pending 锁、409 后重取服务器 job 状态并禁用重复 rerun；`CaptionReview.test.tsx` **2/2 passed**。修复位于 `CaptionReview.tsx`：rerun/cancel 遇 409 后主动 refetch 当前 job，避免终态旧快照继续暴露过期操作。
- 队列单项控制补测：`queue-monitor-mutations.spec.ts` **9/9 passed**；新增 retry 成功/pending 锁、运行项 stop 的目标 URL 与 `delete_runtime:false`、已完成项 remove 仅移出列表的确认和快照断言。首轮新测试因确认文案、筛选和 fixture 计数假设未对齐而失败，修正夹具/断言后整份 spec 通过，未发现对应产品缺陷。
- 历史前端验证快照：r17 完整 mock Playwright 为 **184 项：182 passed、2 skipped、0 failed**；r21 结果仅作为当时记录见文末，不能代表当前最新全量。测试均未连接真实 API/WS。390px dark/light 截图在 `/tmp/dragon-next-e2e-s1-20260925-r15/` 已抽查，未见图表、控件、日志或设备信息遮挡/裁切。
- 契约边界复核：后端 `/api/training/queue/batch/start` 有预检失败 `failed_index`，运行时部分入队则返回 `ok=false`、`queued_count`、`requested_count`、`queued_items`、`failures`；`tests/test_training_queue.py` 已覆盖。`web/frontend-next/src` 与 `e2e` 没有该端点调用，故这是当前 Next 的 N/A，不报告为已覆盖的 Next UI，也不虚构队列批量取消的逐项部分失败语义。蒙版 `apply_masks` 对选定子集修改内存配置后执行单次配置保存，只在成功时返回全部 `indices`，没有每子集 partial-result response；500/网络断开覆盖的是结果待核对/显式重试，不据此推断部分成功。
- 训练启动服务端调用链复核发现，`POST /api/training/start` 的 HTTP 500 可能发生在子进程成功创建之后（`web/services/training/launcher_job.py` 启动子进程后仍执行日志、广播和任务跟踪）；因此状态未知不能当作可安全重试。前端同一弹窗内禁用重试；新增错误态链接可打开当前监控核对服务器快照，关闭/重开仍会重新预检并要求确认。mock E2E 验证了 500/断网时不自动重试、pending 锁和监控导航，但不证明真实 launcher、进程或恢复链路。
- 本轮前端默认并行检查出现分散的 UI 测试等待/超时；唯一在串行全套中重复复现的失败位于 `DatasetWorkspace.test.tsx`：图片 error 事件后同步断言 React 状态。改为 `waitFor` 等待 `data-image-error=true` 提交，不改预览产品逻辑。定向及串行全量验证通过：TypeScript 通过，Vitest 49 文件 / 221 用例通过（`pnpm exec vitest run --maxWorkers=1`）；默认并行运行中的其他超时不作为通过证据。
- 中间完整 Playwright r6（160 项，153 passed、2 skipped、5 failed）暴露 3 项数据集 E2E 把 HTTP 409 错当成可直接重试；源码刻意锁住旧 revision 并要求“重新读取”。测试已改为断言草稿保留、取消 reload 不丢草稿、确认 reload 载入服务器 revision 后才允许重新编辑和保存。另 2 项封面图片与图表 canvas 等待在单 worker 定向及最终全量中通过，未确认产品回归。
- 上轮完整 Playwright r9 收集 164 项，162 passed、2 skipped、0 failed，耗时 8.9 分钟；本轮更新结果见本记录首条。该历史计数包括当时工作树中的 S2/S3 mock specs。
- 懒加载错误用例会在 Vite 控制台记录预期的动态导入失败；React Router 的 `RenderErrorBoundary` 接住异常，用户可见错误恢复页，不是未处理崩溃。
- 原有 `workspace.spec.ts` 已覆盖八个一级工作区直达路由、五种视口、深浅主题及无横向溢出；`interactions.spec.ts` 覆盖部分弹窗 Escape、焦点恢复与焦点约束。生产只读截图基线见 [S0 记录](dragon_next_s0_baseline_20260924.md)。
- 本机人工快速抽查 S0 的 datasets、monitor、history、models、captioning 页面在 1440×900、390×844 和 360×800 下的截图，未见明确文字/控件相互遮挡。部分 datasets/monitor 截图仍处于接口加载中；S0 截图可能含真实配置路径/任务信息，仅在 `/tmp/dragon-next-production-20260924-s0-r2/` 本机查看，未外发。
- history 手机筛选栏最初看起来拥挤；测量确认搜索框修复前 390px 为 54px、360px 为 24px，确认为 P2 响应式问题。`HistoryPage.css` 现让 600px 以下搜索项独占一行；隔离 E2E 断言两种视口输入宽度均 >=240px、无 document 横向溢出，`history-return.spec.ts` 7/7 通过。
- 修复后的 mock 截图 `/tmp/dragon-next-e2e-20260925-s1-history-visual/`（390×800、360×800）已按 UI rubric 人工复核：搜索输入占满首行，筛选控件与状态图例分行且未见遮挡；截图仅含 fixture 数据。最后一次窄屏定向 2/2 通过并生成截图。
- 本轮状态反馈截图抽查 `/tmp/dragon-next-e2e-s1-20260925-state-feedback/`：实际查看 390×844 dark/light 的 monitor、queue，以及 1440×900 dark 的两页；未见标题、状态、主要控件、图表或任务卡遮挡/裁切，长配置路径按省略号展示。此为抽样，不覆盖 768/1280、1440 light，也不代表读屏顺序或完整视觉矩阵通过。

## 发现与未完成项

- skip-link dirty POP blocker 绕过属于 P1，已按上文修复并由训练与数据集隔离 E2E 覆盖；仍需后续常规回归守住 Router hash 与两种 dirty guard 的契约。
- S1 的 70 格 mock 状态矩阵已机器校验并登记；最终退出仍需把真实服务/持久态、人工读屏与视觉边界和剩余组合核验闭合，不能把 mock 证据当作签收。
- 尚未完成所有页面、主题和缩放组合的人工视觉复核；移动 provider 弹窗焦点与 lazy chunk 加载失败恢复已有 mock E2E 证据，真实读屏器顺序仍缺证。减少动态效果有自动化媒体初始值和运行时切换覆盖，仍未完成人工体验检查。
- history 搜索窄屏宽度缺陷已修复并在 mock fixture 下验证；截图基线仍是修复前的生产静态包，不代表修复后生产 UI。
- 真实服务写入、训练与外部 provider：NOT RUN；下表仅概括前端/mock 证据，不能据此推断后端或持久化通过。

### 工作区状态矩阵缺口

本轮把八页状态矩阵的证据登记拆成 10 个 workspace（含图片与蒙版子路由）× 7 个状态，共 70 行；严格区分 `PASS_EVIDENCE`、`NOT_RUN`、`N/A` 和 `MANUAL_REQUIRED`。校验命令：`node web/frontend-next/scripts/validate-s1-state-matrix.mjs`。共享 `mockWorkspace()` 默认成功响应不单独计为通过。

| 工作区 | 已有前端/mock 证据 | 关键未覆盖项 |
| --- | --- | --- |
| 训练配置 | dirty/409 保存失败保留草稿、只读另存、preflight 错误与成功、设备缺失、入队 pending/断网 unknown/无自动重试；立即启动 pending 锁、500/断网后不自动重试、重开弹窗重新预检确认及打开监控核对状态；启动成功完整核对 `config_file`、`preset`、`variant`、`methods_subdir`、确认标志和 GPU 白名单；capabilities 目录 503 锁住执行并在显式重试后恢复；训练量估算 503 错误保持空态、默认重试窗口结束后停止并由“重新估算”恢复；`training-draft-flow.spec.ts`、`training-devices.spec.ts`、`training-estimate.spec.ts` | 真实启动结果与 launcher 状态核验、其他提交失败恢复、长路径；监控链接仅在 mock snapshot 下验证 |
| 数据集蓝图 | dirty 保存失败、保存 409 保留草稿并锁住旧 revision；跨组排序失败回滚；预设库 503→显式重试；CRUD E2E 覆盖新建/保存/另存/重命名/删除、重复提交仅发一次、删除 409 保留已选预设并显式重试；另存并发 409 后保留 dirty 草稿并可重试、慢请求期间锁定操作和表单；重命名创建冲突/响应未知/断网时保留旧预设并支持显式重试，旧文件删除 409 保留新旧预设，删除已提交但响应丢失后按列表快照对账；导入/导出 round-trip、dirty 状态导入保护、409 后保留弹窗与原选择、显式重试后才切换；导入响应丢失后锁住直接重试，关闭弹窗并刷新预设库供人工核对；已保存预设图片工作台往返；390/1440px 长名称/路径无页面溢出且列表路径明确省略；`dataset-crud-mutations.spec.ts`、`dataset-import-export.spec.ts`、`dataset-draft.spec.ts`、`dataset-drag.spec.ts` | 导入响应丢失后的刷新只提供服务器列表证据，尚未自动判定同名内容是否一致；70 格已登记，但真实/人工签收仍未闭合 |
| 图片/蒙版/打标子路由 | 三视图 URL 上下文、有效图片预览完整响应与零运行时错误、部分预览失败、缺 dataset 的预览/蒙版深链空态与返回；蒙版列表/预设 GET 503 错误态及显式重试恢复；蒙版 dirty 离页及切图/切子集取消/确认、保存 409 保留草稿并用原 revision 重试成功、应用到子集 busy/409/500/断网后显式重试；r91 追加第二子集 preview/mask 深链参数和根切换 pending 撤旧数据；Next 打标 TXT commit 409/断网不自动重试及候选保存断网后保留/重试；`mask-workspace.spec.ts`、`dataset-media-root-isolation.spec.ts`、`DatasetImageWorkspacePage.test.tsx`、`dataset-cover.spec.ts`、`caption-context.spec.ts` | apply 请求响应丢失后的磁盘持久态需用隔离 HTTP/临时根核对；当前接口没有逐子集部分结果契约；source=training、非零 offset/limit 和其他图片参数组合仍未穷尽 |
| 训练队列 | loading/error/cached/empty、bulk revision 409 与缺 revision 防护、move HTTP 409/503 保留旧快照/解锁/无自动重试、move/retry/remove unknown 刷新对账、cancel/retry 404 对账、retry 503 保留 error 快照、running stop 404/503 保留旧快照且显式刷新核对、`delete_runtime:false`、stop/remove 响应丢失后手动刷新收敛、done remove 成功；策略 mutation pending/500/断网覆盖；`queue-monitor-mutations.spec.ts`、`queue-policy-mutations.spec.ts`、`queue-scope.spec.ts` | 真实队列服务和隔离持久态仍未核验；batch-start 部分入队 API 当前没有 Next 调用入口，列为范围外 |
| 当前监控 | 局部 HTTP 失败/重试、旧 task WS 过滤、停止目标/409、无任务/未知状态/GPU empty、GPU 503 显式重试恢复、WS 连续失败退避与成功复位、任务与连接状态 polite status、图表 4 视口 × 2 主题边界；日志滚动区可键盘聚焦并有可访问名称；`state-feedback.spec.ts`、`queue-monitor-mutations.spec.ts`、`useWebSocket.test.tsx`、`LogViewer.test.tsx` | 真实读屏顺序及连续重连状态的人工核验 |
| 历史任务 | 列表 503→恢复重试→确认空态、日志失败重试/空搜索、深链返回、archive HTTP 409/503 保留任务与选择、pending 锁/无自动重试、刷新核对后清错；delete 成功/响应丢失对账；续训 409→重新检查→显式重试、已接受入队但响应丢失后锁定并按队列快照核对；resume-options 503 时禁用提交并在显式重读后恢复；optimizer 缺失项与可用项并存时切换会清除旧确认，只提交重新确认的有效路径；artifact manifest available/missing/blocked/unreadable、样张目录缺失、概览和详情样张/权重独立 503 与显式重试；搜索布局修复；`history-actions.spec.ts`、`final-audit.spec.ts`、`history-overview-assets.spec.ts`、`s3-queue-monitor-history.spec.ts`、`state-feedback.spec.ts`、`history-logs.spec.ts`、`history-return.spec.ts` | 其他完整性字段的组合仍未穷尽；真实 launcher、真实 history/output 持久态不由 mock 验证 |
| 模型配置/全局设置 | 读取失败重试、模型 revision 409 确认重载、默认项保护/切换、保存 pending 锁与成功同步、模型 PUT 断网 unknown 后显式核对、排序 DOM 与 PUT 顺序一致及长路径；设置根切换后训练配置、数据集、模型、历史、队列代表性 query 均读取对应新根；新根 dataset GET 503 时无旧缓存泄漏并可显式重试；全局设置 PUT 断网 unknown 保留 patch，普通成功响应同步回表单并禁用保存；`s2-isolation.spec.ts`、`model-library-order-path.spec.ts`、`state-feedback.spec.ts` | 尚未逐项枚举每个工作区和非根相关 query key；空模型列表不是有效后端响应（GET/PUT 均拒绝） |
| 打标工作台 | 跨页 draft 身份与候选选择、dirty 防丢、missing image/retry、候选 PATCH 断网后保留草稿并手动重试、TXT commit 409/断网后不自动重试并可显式重试、provider key 不回显、390px 弹窗焦点、provider ping 502/断网显式重试与 pending 锁、job rerun pending/409/500/未知结果核对；运行中 job cancel 确认/pending 锁/成功、409 刷新终态、500 保留 running 快照并可显式重试、未知结果锁定并核对；r69 另覆盖配置根切换后 source preset 与默认图片页重读；r91 覆盖 model-assets/tag-dictionary 连续读取失败后的显式刷新恢复；提示词预设页和图片来源页在 prompt-presets 失败后保持错误、停止自动重试并由“重试提示词”恢复；`caption-context.spec.ts`、`interactions.spec.ts`、`captioning-root-isolation.spec.ts`、`captioning-assets-states.spec.ts`、`captioning-prompts-states.spec.ts` | profiles/settings/secrets/jobs/detail/logs 等 query 的完整错误/恢复和存储根组合仍未穷尽；读屏器与 live-region 实际朗读顺序仍未人工核验 |

该表是下一轮定向补测清单，不把未覆盖项等同于已发现产品缺陷。真实 HTTP 隔离根写入属于 S2；S1 仍需优先补足各页错误/空/busy/成功/冲突/断网结果状态、读屏顺序和多主题/缩放人工复核。

### Query Key 对账边界

根切换本身不编码进大多数 query key；全局设置返回 `requires_reload` 时会移除并重新失效所有首段不是 `settings` 的 query，随后清空训练配置选择（`SettingsPage.tsx`）。r44 的 `s2-isolation.spec.ts` 只证明训练配置 groups/presets/merged、数据集库、模型、历史列表和队列快照这些代表请求按新根读取，不等于穷尽所有 workspace key。

截至 r77，根切换行为证据覆盖训练 raw TOML 同 key 重读、训练 estimate 同 key 重读、training dataset picker preview 及所选文件的 preflight POST；数据集 library/cover/preview、蒙版 offset=0/48 列表及图片请求；history list/detail/artifact manifest、样张/权重摘要、resume-options，以及日志 metadata 和 `offset=400&limit=400` 的代表性分页；captioning 来源上下文的 dataset library/preset/image `offset=0/60`。r69/r70 使用同一 dataset file；新根请求 pending 时旧来源/图片结果不显示，URL/query 参数保持相同。resume-options 新根响应挂起时旧 checkpoint 路径不再是选中值；日志 metadata 挂起时旧 800 行快照和日志行不显示，放行后读取新根 500 行与 offset 400 正文；captioning 新根 `offset=60` 页请求挂起时第一页图片撤下，放行后显示新根第 61 张图片。未触发续训或其他写入。

仍未逐项核验的 query/请求路径包括：数据集 preview 的 `source=training`、显式非零 offset/limit 和更多子集组合；mask 的非零 offset、其他 image 文件及更多参数组合；history 除 search 外的其他 offset/limit 组合。history collections 的后端文件通过动态 `HISTORY_DIR / collections.json` 随 `history_root` 解析；r82 mock 下前端固定 query key 在设置切根后重新 GET，集合按钮由旧根 `history-only` 切换到新根 `external-only`，未执行集合写入。r84 还验证 search 请求携带 `query/cursor/direction`，新根 pending 时旧匹配不显示，放行后结果计数和匹配行来自新根。r91 的 Captioning assets/dictionary 已覆盖连续 GET 失败、无自动重试和显式“刷新状态”恢复；profiles/settings/secrets/prompt presets 随进程启动时的 `ANIMA_HOME`，assets/dictionary 随专用环境变量或 `ANIMA_HOME`；jobs/detail/logs 是进程内状态，不应断言它们随 `configs_root` 隔离。captioning job 引用的 dataset/image 路径须另按 dataset resolver 检查。Preflight 已验证为切根后的显式 POST，但不是 query cache key；不等于启动/入队执行已验证。此处仍是覆盖缺口，不是已确认缓存泄漏；真实持久态仍归 S2。

### S1 续审更新（2026-09-26，r78-r80）

- 数据集导入补充 409 和 POST 响应丢失 mock E2E。409 时 pending 禁止重复提交、修改名称、取消或 Escape 关闭；弹窗、源名称和原选中预设保留。等待后无自动重试，用户再次提交才切到导入结果。响应丢失时原 UI 虽显示结果未知却允许盲重发，现禁用直接重试，提供“关闭并刷新预设库核对”；fixture 模拟服务端已提交后断开响应，刷新后列表出现新预设而 POST 仍只有一次。
- 完整 `dataset-import-export.spec.ts` **5/5 passed**，最终输出 `/tmp/dragon-next-e2e-s1-dataset-import-r80/`；首轮 r78 的一处失败是断言误用完整文件名，实际列表将 stem 与路径分开显示，失败快照中已出现新预设。r79 修正断言后通过，r80 加强 pending 锁后再次通过。TypeScript 通过；单 worker Vitest **53 文件 / 231 用例通过**。
- 测试使用隔离 Vite 端口 `20963`、不可达 API target 和 route mocks，没有访问真实配置根、provider、launcher、队列或持久数据。只验证刷新列表供用户核对，不把同名文件的内容一致性或真实 HTTP 写入推断为已证。

### S1 续审更新（2026-09-26，r81-r82）

- 在原 history 根切换 fixture 增加 root-specific `collection_order`：旧根 `history-only`，新根 `external-only`。跨根导航后检查新根 GET、旧集合按钮消失和新集合按钮出现，未发送集合 PUT。`s2-isolation.spec.ts` 定向 1/1、完整 spec **9/9 passed**，输出 `/tmp/dragon-next-e2e-s1-history-collections-r82/`。
- 这证明 mock 下设置 `requires_reload` 后固定 collections query key 可重新读取并刷新 UI；后端 `<history_root>/collections.json` 动态路径仅由源码核对，真实根切换/持久态不由本 E2E 证明。history log search/其他分页仍待核验。

### S1 续审更新（2026-09-26，r84-r85）

- `history-root-log-isolation.spec.ts` 增加根切换后的全局日志搜索：旧根响应 `1 / 2`，新根请求使用 `query=root needle&cursor=0&direction=forward`；新根搜索挂起时显示“搜索中…”且没有旧匹配，放行后日志跳到新根第 500 行并显示 `2 / 2`。完整定向 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-history-log-search-r84/`；无写入、未连接真实 history 服务。
- 修复蒙版缩放状态的可访问性：`MaskTools` 的百分比输出现在带 `aria-live="polite"` 与 `画布缩放` 标签；新增 `MaskTools.test.tsx` 验证 Enter/点击缩放命令与 100%→125%→100% 状态同步，定向 Vitest **1 文件 / 1 用例 passed**，TypeScript 通过。该修复不改变 canvas 缩放算法。
- 上轮完整 Playwright r83 在 265 项中运行到 258 项后因会话中断且未生成报告目录，不能记作通过；r86 完成但暴露 1 个监控错误态按钮卸载失败（262 passed/2 skipped/1 failed），单 worker r87 与完整 monitor spec r89 均通过。r90 修复后取得 **263 passed/2 skipped/0 failed** 的完整终态。S1 仍未签收，真实读屏器/人工视觉、真实服务和持久态边界不变。

### S1 续审更新（2026-09-27，r86-r90）

- r86 全量暴露可复现时序问题：监控状态错误后仍以 2 秒间隔自动轮询，服务恢复时会在用户点击前卸载“重试任务状态”按钮，导致 `monitor-layout.spec.ts` 点击超时。定向单 worker r87 通过但不能掩盖并行全量失败。
- 修复 `LiveMonitorPage`：状态查询出现错误时停止 `refetchInterval`，恢复必须由用户显式点击“重试任务状态”；E2E 新增 5.5 秒无自动状态请求断言，并在显式重试后核对请求恢复。完整 `monitor-layout.spec.ts` **11/11 passed**（r89）。
- r90 完整 mock API/WS Playwright **265 项，263 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-20260927-r90/`，耗时约 9.5 分钟。新增数据集导入、history log search、蒙版缩放状态测试均包含在全量中；API/WS 仍由 mocks 接管。
- 单 worker Vitest **54 文件 / 234 用例通过**；TypeScript、文档完整性 **8/8**、`rtk git diff --check` 通过。两份共享工作树新 Qwen findings 补充生命周期 `状态：` 行后，文档门禁恢复全绿；未回滚其内容。

### S1 续审更新（2026-09-27，r91）

- 新增 `dataset-media-root-isolation.spec.ts` 两项 mock-only E2E：第二子集深链 preview/mask 使用 `dataset_index=1`；preview 保持 `source=source&limit=120`，mask 列表保持 `offset=0`，蒙版图片请求携带第二子集的完整 image 路径。切换 `configs_root` 后，新根请求 pending 期间旧目录、caption、mask 目录和 canvas 均撤下，放行后只显示新根数据；`mocks.writes` 与 `mocks.unhandled` 为空。
- 新增 `captioning-assets-states.spec.ts`：`/api/captioning/model-assets` 与 `/api/captioning/tag-dictionary` 连续失败后，页面保持可见错误且没有资源/词典旧快照；等待默认查询重试窗口后请求数不再增加，用户显式点击“刷新状态”才恢复。没有触发下载、provider 或其他写请求。
- r91 定向命令使用隔离 Vite 端口 `20975`、不可达 `127.0.0.1:29999` target，结果 **3/3 passed**，产物 `/tmp/dragon-next-e2e-s1-r91d/`；TypeScript 通过，测试文件共 495 行，未修改后端逻辑或用户数据。
- S1 仍为 **NOT RUN / 未签收**：source/offset/limit 的其余组合、八页完整状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实服务、launcher 和隔离持久态继续未完成。

### S1 续审更新（2026-09-27，r92）

- 完整 mock API/WS Playwright **268 项，266 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-r92/`，耗时约 16.7 分钟；2 个 skip 仍是 `dataset-drag.spec.ts` 的折叠分组 hover/drop 场景。r91 的第二子集媒体隔离和 Captioning 资源/词典恢复用例已纳入全量。
- 单 worker Vitest **56 文件 / 241 用例通过**；TypeScript、文档完整性 **8/8**、`rtk git diff --check` 通过。完整运行仍只使用隔离 Vite、API/WS route mocks 和不可达 target，未触碰真实 launcher、provider、用户目录或持久态。
- S1 仍为 **NOT RUN / 未签收**：source/offset/limit 的其余组合、八页完整状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实服务、launcher 和隔离持久态继续未完成。

### S1 续审更新（2026-09-27，r93）

- 修复 `LogViewer.tsx` 的滚动日志区域：`role="log"` 现在带 `aria-label="日志内容"` 与 `tabIndex=0`，键盘用户可以聚焦并滚动超出区域；`LogViewer.test.tsx` 验证可访问名称和焦点行为。`CommandDialog.tsx` 的关闭按钮与 Escape 现在共用 busy 守卫，`CommandDialog.test.tsx` 验证 busy 时不关闭、空闲时可关闭。
- 修复 Captioning prompt 读取失败的局部恢复：`CaptionPrompts.tsx` 与 `CaptionSource.tsx` 分别提供“重试提示词”，错误期间不伪造预设列表；`captioning-prompts-states.spec.ts` 覆盖提示词页和图片来源页，连续 503 在默认重试窗口后停止，显式重试后只增加一次 GET 并恢复 `Studio captions`。两条用例均断言无 fixture writes/未处理请求。
- 为训练量估算增加 mock-only 失败证据：`training-estimate.spec.ts` 验证 503 后只显示 alert/空态，等待窗口内请求不再增加，点击“重新估算”后恢复分桶表且只增加一次 GET；该用例不改变现有自动重试策略。
- 新增机器可核验状态登记：[dragon_next_s1_state_matrix_r93.json](dragon_next_s1_state_matrix_r93.json) 记录 70 个单元格；`node web/frontend-next/scripts/validate-s1-state-matrix.mjs` 输出 `S1 state matrix valid: 70 rows, overall_state=NOT_RUN`。它明确列出当前未运行与需人工检查的格子，不把文件存在或默认 fixture 当作通过。
- r93 完整 mock API/WS Playwright **271 项，269 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-r93/`，耗时约 16.8 分钟；单 worker Vitest **58 文件 / 244 用例通过**，TypeScript 通过。只使用隔离 Vite、API/WS route mocks 和不可达 target，未触碰真实 launcher、provider、用户目录或持久态。
- S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项机器化对账，真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实服务、launcher 和隔离持久态继续未完成。

### S1 续审更新（2026-09-27，r94）

- 训练配置保存冲突新增 mock-only 浏览器证据：首次 PATCH 返回 409 后，`local-draft` 仍保留，保存按钮禁用，且请求次数严格为 1；显式确认“重新加载配置”后读取 `server-v2` 与 `server-revision`，reload 过程不发送 PATCH；用户再次编辑后第二次 PATCH 使用 `server-v2`，成功响应更新为 `server-v3`。
- `training-draft-flow.spec.ts` 完整定向 **6/6 passed**；命令使用隔离 Vite 端口 `5189`、不可达 API target `127.0.0.1:29999`，产物 `/tmp/dragon-next-e2e-s1-training-config-409-r94/`。该结果只证明前端在 mock API/WS 协议下的状态语义，不证明真实配置服务、磁盘 revision 或持久化。
- 机器矩阵保持 70 行、`overall_state=NOT_RUN`；`training.conflict_409` 由 `NOT_RUN` 更新为 `PASS_EVIDENCE`，其余未运行/人工格子不作推断。r93 的 271 项完整套件计数仍是最近一次全量基线，本轮不外推全量通过数。

S1 仍为 **NOT RUN / 未签收**：真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，真实服务/launcher/provider、持久态与发布仍未执行；状态矩阵仍有未运行和人工必需单元格。

### S1 续审更新（2026-09-27，r95）

- 图片工作台的预览 GET 失败新增 mock-only 浏览器证据：持续 503 时显示“无法读取数据集预览”和服务端错误，不显示旧/伪造图片卡片；默认 query 重试结束后请求停止，用户点击“重试”才增加一次读取并恢复 3 张图片。完整 `mask-workspace.spec.ts` **14/14 passed**，无 fixture writes/未处理请求。
- 本轮使用隔离 Vite 端口 `5190`、不可达 API target `127.0.0.1:29999`，产物 `/tmp/dragon-next-e2e-s1-dataset-image-error-r95/`；不连接真实数据集服务或持久目录。`offline_unknown` 对只读 GET 没有提交后结果核对语义，已明确登记为 `N/A`，没有伪造测试证据。
- r93 的 271 项完整 Playwright 仍是最近一次全量基线；r94/r95 仅为定向增量，不外推全量通过数。矩阵仍保持 `overall_state=NOT_RUN`。

S1 仍为 **NOT RUN / 未签收**：真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，真实服务/launcher/provider、持久态与发布仍未执行。

### S1 续审更新（2026-09-27，r96）

- Captioning 的 confirmed-empty 状态新增 mock-only 浏览器证据：`/api/captioning/jobs` 返回空数组时显示“暂无打标任务”，同时“图片来源”和“新任务”入口保持可用；没有创建 job、调用 provider 或产生 fixture writes。`captioning-prompts-states.spec.ts` 完整定向 **3/3 passed**。
- 命令使用隔离 Vite 端口 `5191`、不可达 API target `127.0.0.1:29999`，产物 `/tmp/dragon-next-e2e-s1-captioning-empty-r96/`。这是 Next mock API/WS 证据，不等价于真实 captioning 服务或 provider 验收。
- 机器矩阵仍为 70 行、`overall_state=NOT_RUN`；`captioning.empty` 已由 `NOT_RUN` 更新为 `PASS_EVIDENCE`。r93 的 271 项完整套件计数仍是最近一次全量基线，r94-r96 仅为定向增量。

S1 仍为 **NOT RUN / 未签收**：14 个矩阵格子仍需人工核验，5 个格子按契约为 `N/A`；真实读屏器/视觉、服务/launcher/provider、持久态和发布仍未执行。

### S1 续审更新（2026-09-27，r97-r98）

- r97 的设置/模型状态反馈 mock E2E 已补齐读取 gate、保存 pending 锁和成功同步默认项：`state-feedback.spec.ts` **23/23 passed**，产物 `/tmp/dragon-next-e2e-s1-settings-model-loading-r97/`。只验证前端状态语义，不证明真实 settings/model service 或磁盘。
- r98 新增三个初始读取 gate：数据集库预设 GET 挂起时保持 `aria-busy` 与“正在读取预设库”；蒙版列表响应后再挂起图片 GET，期间画布、保存和应用命令不可用；监控先挂起 status，再挂起依赖的 metrics/logs/GPU，未出现陈旧确认内容。`training-dataset-picker.spec.ts`、`mask-workspace.spec.ts`、`monitor-layout.spec.ts` 合计 **33/33 passed**，命令使用隔离 Vite 端口 `5192`、不可达 API target `127.0.0.1:29999`，产物 `/tmp/dragon-next-e2e-s1-loading-r98/`。
- 矩阵内部快照更新为 `r98`：70 行中 **59 `PASS_EVIDENCE`、6 `MANUAL_REQUIRED`、5 `N/A`**；`overall_state` 保持 `NOT_RUN`。新增证据只来自 mock API/WS，未连接真实服务、launcher、provider、用户目录或持久态。

S1 仍为 **NOT RUN / 未签收**：剩余 6 个人工格子、5 个契约 `N/A`，真实读屏器/视觉、服务/launcher/provider、持久态和发布仍未执行。

### S1 续审更新（2026-09-27，r99-r102）

- r99 的训练/历史增量合计 **22/22 passed**，产物 `/tmp/dragon-next-e2e-s1-training-history-r99/`：训练 loading/empty/success 只验证配置上下文与命令门禁；历史 success 只验证 mock task 的摘要、产物和日志 identity。
- Captioning loading/success 已补齐。r100 的 success 用例通过；同轮 loading 初次使用了桌面隐藏任务切换按钮的错误 locator，修正后 r102 定向 **1/1 passed**，产物 `/tmp/dragon-next-e2e-s1-captioning-loading-r102/`。本组只使用 route mocks，没有 provider、真实 TXT 或持久化写入。
- 矩阵内部快照更新为 `r102`：70 行中 **65 `PASS_EVIDENCE`、0 `MANUAL_REQUIRED`、5 `N/A`**；`overall_state` 仍为 `NOT_RUN`。清零人工格子只代表 mock 状态矩阵完成，不代表真实服务、读屏器、视觉、持久态或发布签收。

S1 仍为 **NOT RUN / 未签收**：5 个契约 `N/A` 之外，真实读屏器/视觉、服务/launcher/provider、持久态和发布仍未执行。

## S1 续审更新（2026-09-26，r51-r53）

- 全局设置新增普通 PUT 成功路径：提交 `{ ui_scale: 125 }`，mock 服务端返回规范化 `ui_scale: 130`，验证表单采用响应值、提示“全局设置已保存”、状态为“已同步”、保存按钮禁用且只发送一次 PUT。完整 `state-feedback.spec.ts` **19/19 passed**，输出 `/tmp/dragon-next-e2e-s1-settings-success-r51/`。
- 训练立即启动成功路径现在核对完整请求体：`config_file`、`preset`、`variant`、`methods_subdir`、`confirmed`、`confirm_preprocess`、`gpu_whitelist` 均与当前选择一致。完整 `training-devices.spec.ts` **7/7 passed**，输出 `/tmp/dragon-next-e2e-s1-training-target-r52/`。
- 完整 mock API/WS Playwright r53 **239 项，237 passed、2 skipped、0 failed**，2 workers、8.8 分钟，输出 `/tmp/dragon-next-e2e-s1-20260926-r53/`；两个 skip 是 `dataset-drag.spec.ts` 的折叠分组 hover/drop。命令：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20730 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r53`。全量包含 r51/r52 新增场景。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**；文档完整性 **8/8 passed**、`rtk git diff --check` 通过。E2E 使用隔离 Vite 端口 `20728`-`20730`、不可达 API target `127.0.0.1:29999` 和 route mocks；未修改产品逻辑，没有连接真实 API/WS、launcher/provider 或用户持久数据。
- 工作树仍为 `dev` / HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`、ahead 13，含大量既存修改。本轮只修改 `state-feedback.spec.ts`、`training-devices.spec.ts` 和两份 S1 findings；静态包未构建或替换，没有清理、暂存或提交。
- Query key 对账将 r44 的覆盖边界明确收窄到代表性 query；training raw/estimate/preflight/picker-preview、dataset preset/image/cover/mask、history collections/detail/assets/summary、captioning 查询仍未与根切换组合验证。未确认产品缺陷。

S1 仍为 **NOT RUN / 未签收**：完整八页状态矩阵与 query key 根切换矩阵尚未逐项关闭；真实读屏器/live-region 朗读顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r54-r59）

- r56 `s2-isolation.spec.ts` **5/5 passed**，补充设置配置根切换后的蒙版 preset、offset=0 列表和相同图片参数重新读取；列表与图片请求分别 gate，确认旧根目录/canvas 不残留。r54/r55 的失败来自 `aria-busy`/未命名 status 定位器假设，修正后未发现产品缺陷。
- r58 补 cover/preview 的 query 根切换：pending 时旧封面、目录和摘要撤下，放行后显示新根内容；完整 spec **6/6 passed**。
- r59 是最近一次完整 mock API/WS Playwright：**241 项，239 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-20260926-r59/`。`web-next-check` 为 TypeScript 通过、Vitest 50 文件 / 226 用例；API/WS 由 mocks 接管。
- 这些结果仍是隔离浏览器证据；未连接真实 API/WS、launcher、provider 或用户持久数据，也未构建/替换静态包。

S1 仍为 **NOT RUN / 未签收**：training raw/estimate/preflight/picker-preview、非零 mask offset、history 和 captioning 的 query 根切换尚未逐项核验；人工及真实服务边界不变。

### S1 续审更新（2026-09-26，r60-r64）

- r60 在 `s2-isolation.spec.ts` 增加训练 raw TOML 根切换 mock E2E：新旧根返回不同 raw 内容，但始终使用 `configs/imported/studio-portrait.toml`；切换后经 SPA 导航重新选择同一文件，断言 raw GET 仍带相同 `file`、显示新根内容，设置 PUT 仅一次且无其他写入。所在 spec 当时 **7/7 passed**，输出 `/tmp/dragon-next-e2e-s1-training-raw-r60/`。
- r61-r62 扩展蒙版 fixture 分页，并验证 offset 48 页面跨根重新读取。初始尝试暴露 locator 问题：工作台列表已默认打开且 toggle 不在可访问树；其后全页旧名称查询也不稳定。失败快照中的当前列表已显示新根内容。改为直接使用分页命令并检查 `.mask-images .mask-image-item` 当前名称后，定向用例 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-mask-offset-r63/`；未发现或修改产品逻辑。
- 最终 `s2-isolation.spec.ts` **8/8 passed**，输出 `/tmp/dragon-next-e2e-s1-root-training-mask-r64/`，隔离 Vite 端口 `20741`、不可达 API target `127.0.0.1:29999`、mock API/WS。包含 raw、offset 48、默认 offset 0、cover/preview 和设置根切换错误恢复场景。`web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。
- r59 的完整 mock Playwright **241 项，239 passed、2 skipped** 仍是最近一次完整 suite；本轮没有重跑全量，不把新用例计入 r59。文档完整性 **8/8 passed**；普通 `rtk git diff --check` 无输出，但本次目标文件为未跟踪文件，另以 `--no-index --check` 检查它们。未连接真实 API/WS、launcher、provider 或用户持久数据；未构建/替换静态包、暂存、提交或清理工作树。

S1 仍为 **NOT RUN / 未签收**：training estimate/preflight/picker-preview；数据集其他预览参数与 mask-image 组合；history collections/resume/log pages；captioning 的 profiles/jobs/logs/prompts/assets/dictionary 等查询仍未逐项核验。完整八页状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工验收，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r65-r70）

- r65-r68 在同一 history task ID 上验证切换 history root 后重新读取列表、详情、artifact manifest、样张与权重摘要；新根标题、配置 artifact、样张和权重结果均可见。首轮测试因历史分组默认折叠及 StrictMode 重复 GET 假设失败，改用可见交互与最终根响应断言后，完整 `s2-isolation.spec.ts` **9/9 passed**，输出 `/tmp/dragon-next-e2e-s1-root-query-r68/`；未发现产品缺陷。
- r69 新增独立 `captioning-root-isolation.spec.ts`：切换 `configs_root` 后，在打标来源对同一 dataset file 重读 preset 与默认图片页；新根请求 pending 时旧图片/目录不显示，完成后新根内容出现。文件、dataset index、source、limit、offset 均保持一致。首轮仅因 mock PUT 断言漏了 revision 字段失败，修正后 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-caption-root-r69b/`。
- r70 新增 `training-root-isolation.spec.ts`：在同一 `config_file` 下，切根后重新读取 estimate 和 dataset picker preview；gate 新根响应时断言旧估算路径/缩略图不显示，放行后新根内容出现。另运行 preflight mock POST，确认所选 config path、variant/preset/methods_subdir 保持绑定；没有启动或入队。前两次失败是未打开估算弹窗、以及严格要求 preset GET 恰好两次；按真实交互和最终新根响应修正后 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-training-root-r70/`。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20838 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/training-root-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-training-root-r70`。r68/r69 命令见前一条。前端门禁命令：`rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check`。
- `web-next-check` 通过：TypeScript，Vitest **50 文件 / 226 用例**。r68-r70 使用隔离 Vite、route mocks 和不可达 API target；没有产品逻辑修改、真实 API/WS/launcher/provider 调用、用户数据写入或静态包发布。
- 最近一次完整 suite 仍为 r59（241 项，239 passed、2 skipped）；未重跑全量，r68-r70 不计入该数字。工作树保留 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`、ahead 13 的既存状态，无暂存/提交/清理操作。

当前 root-switch 覆盖包含：training raw/estimate/preflight/picker-preview；dataset library/cover/preview 与蒙版 offset 0/48/list-image 请求；history list/detail/artifacts/images/weights summary；captioning 来源 dataset library/preset/default images。尚缺数据集其他预览参数与 mask-image 组合、history collections/resume-options/log pagination、captioning profiles/jobs/detail/logs/prompts/assets/dictionary 和非零 source-image offset。它们是未覆盖证据，不是已确认缺陷。完整八页状态矩阵、实体读屏器/live-region 顺序、剩余视口/主题/缩放人工验收、真实 launcher/服务及隔离持久态均未完成，故 S1 仍 **NOT RUN / 未签收**。

## 命令

- 当前完整套件 r90：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20972 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260927-r90`；265 项中 263 passed、2 skipped、0 failed，约 9.5 分钟。r86 的 1 个监控失败由 r90 修复后回归关闭。
- 最近一次完整套件 r59：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20736 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r59`；241 项中 239 passed、2 skipped、0 failed，8.8 分钟。r68/r69 是其后的定向补测，不计入全量结果。
- 历史完整套件 r17：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=21429 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-r17`；184 项中 182 passed、2 skipped、0 failed，13.4 分钟。
- 历史完整套件 r15：`rtk proxy timeout 600 env DRAGON_E2E_PORT=20698 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-r15`；175 项中 173 passed、2 skipped、0 failed，8.2 分钟。
- 全量尝试 r16：`rtk proxy timeout 600 env DRAGON_E2E_PORT=20705 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-r16`；收集 178 项，外层 600 秒超时于 174/178，未完成且不能记作通过。当时两项报告失败：light 主题样式用例未等到 monitor canvas，脏蒙版切图用例等待打标 tab 超过 45 秒；改为 single worker 各自复跑均 1/1 通过（`/tmp/dragon-next-e2e-s1-flake-chart-r1/`、`/tmp/dragon-next-e2e-s1-flake-mask-r1/`）。该次运行同时出现一次 `DatasetPreviewDialog` 读取 `row.source_dir` 的运行时错误；已修复通用 preview mock 的缺字段并由新增 preview E2E 验证零 `pageerror`，但这不证明它解释了 mask 的并行超时。r16 本身仍是未完成运行；后续 r17 已完成当前 184 项全量 mock 回归。
- Caption rerun 定向：`rtk proxy env DRAGON_E2E_PORT=21425 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/caption-context.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-caption-rerun-20260925`；15 passed；`CaptionReview.test.tsx` 2/2 passed。
- 队列单项 mutation 定向：`rtk proxy env DRAGON_E2E_PORT=21428 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/queue-monitor-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-queue-mutations-20260925-r3`；9 passed，覆盖 move/cancel 既有场景及新增 retry/stop/remove。
- 历史前端门禁快照：`rtk proxy pnpm --dir web/frontend-next typecheck` 通过；`rtk proxy timeout 300 pnpm --dir web/frontend-next exec vitest run --maxWorkers=1` 通过，50 文件 / 225 用例；当前门禁见文末 r51-r53。
- 本轮完整蒙版工作区定向：`rtk proxy env DRAGON_E2E_PORT=20709 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-mask-regression-r2`；9 passed，包含有效 preview 契约/零运行时错误和脏态切换场景。
- 模型 mutation 锁定定向：`rtk proxy env DRAGON_E2E_PORT=20708 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/state-feedback.spec.ts --grep 'model revision conflict|model library protects its default|model save with a lost response' --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-model-mutation-r3`；3 passed。
- 全量 r16 两项单 worker 复核：`library-style-contracts.spec.ts` light monitor 与 `mask-workspace.spec.ts` 脏态切换各 1/1 passed，命令输出分别见 `/tmp/dragon-next-e2e-s1-flake-chart-r1/`、`/tmp/dragon-next-e2e-s1-flake-mask-r1/`。
- 本轮监控定向：`rtk proxy env DRAGON_E2E_PORT=20695 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/queue-monitor-mutations.spec.ts e2e/state-feedback.spec.ts --grep 'monitor recovers GPU details|monitor and queue layout' --reporter=line --output=/tmp/dragon-next-e2e-s1-monitor-r14`；9 passed，覆盖 GPU 503 恢复及 4 视口 × 2 主题图表边界。
- 本轮队列单项冲突定向：`rtk proxy env DRAGON_E2E_PORT=20697 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/queue-monitor-mutations.spec.ts --grep 'queue cancel reconciles' --reporter=line --output=/tmp/dragon-next-e2e-s1-queue-conflict-r1`；1 passed。
- 本轮 provider 状态恢复定向：`rtk proxy env DRAGON_E2E_PORT=20702 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/interactions.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-caption-provider-r2`；9 passed，含取消确认零请求、502 超时可见、pending 锁、无自动重试及显式重试成功。
- 本轮模型配置状态定向：`rtk proxy env DRAGON_E2E_PORT=20704 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/state-feedback.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-model-config-r2`；17 passed，含默认项保护/切换、保存 pending/成功、409 与断网结果未知后显式核对服务器。
- 本轮前端静态/单元检查：`rtk proxy pnpm --dir web/frontend-next typecheck` 通过；`rtk proxy timeout 300 pnpm --dir web/frontend-next exec vitest run --maxWorkers=1` 通过，50 文件 / 223 用例。新增 `src/app/useWebSocket.test.tsx` 定向 2/2 通过。
- 蒙版工作台定向：`rtk proxy env DRAGON_E2E_PORT=20656 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-mask-r3`；3 passed。
- 数据集冲突恢复定向：`rtk proxy env DRAGON_E2E_PORT=20659 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-draft.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-dataset-draft-r3`；3 passed。
- 壳层与 dirty 数据集定向：`rtk proxy env DRAGON_E2E_PORT=20635 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/app-shell.spec.ts e2e/dataset-draft.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-20260925-s1-guards-final`；7 passed。
- 壳层定向（添加 lazy failure 前）：`rtk proxy env DRAGON_E2E_PORT=20637 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/app-shell.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-20260925-s1-shell-motion`；6 passed。新增的 lazy failure 用例与全部壳层用例均已包含在最新完整套件中。
- history 深链/窄屏定向：`rtk proxy env DRAGON_E2E_PORT=20639 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/history-return.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-20260925-s1-history-search-final`；7 passed，含 390/360px 搜索框宽度与无溢出断言。
- post-change 窄屏截图：`rtk proxy env DRAGON_E2E_PORT=20640 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/history-return.spec.ts --grep "history search remains usable" --reporter=line --output=/tmp/dragon-next-e2e-20260925-s1-history-visual`；2 passed，截图仅含 mock fixture。
- 本轮新增蒙版/数据集用例定向：`rtk proxy env DRAGON_E2E_PORT=20691 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts e2e/training-dataset-picker.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-r11`；13 passed。
- 队列调度策略定向：`rtk proxy env DRAGON_E2E_PORT=20693 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/queue-policy-mutations.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-queue-policy-r1`；4 passed。
- 前端静态与单元检查：`rtk proxy pnpm --dir web/frontend-next typecheck` 通过；默认并行 Vitest 有间歇 UI 等待超时。本轮串行全量命令 `rtk proxy timeout 300 pnpm --dir web/frontend-next exec vitest run --maxWorkers=1` 通过，49 文件 / 221 用例。
- 本轮新增状态/焦点/懒加载用例定向：`rtk proxy env DRAGON_E2E_PORT=20643 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/app-shell.spec.ts e2e/dataset-draft.spec.ts e2e/interactions.spec.ts e2e/state-feedback.spec.ts --grep 'lazy workspace failure|dataset save conflict|history list retries|caption provider dialog receives focus' --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-r5`；4 passed。provider 弹窗截图同目录。
- 设置与历史状态回归：`rtk proxy env DRAGON_E2E_PORT=20646 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/state-feedback.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-state-feedback`；15 passed。
- 队列/监控 mutation 回归：`rtk proxy env DRAGON_E2E_PORT=20648 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/queue-monitor-mutations.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-queue-monitor-mutations-r2`；4 passed。
- 本轮 caption/续训/入队补测：`rtk proxy env DRAGON_E2E_PORT=20651 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/caption-context.spec.ts e2e/overview-states.spec.ts e2e/training-draft-flow.spec.ts --grep 'candidate save keeps|history resume conflict requires|training enqueue holds pending state' --reporter=line --output=/tmp/dragon-next-e2e-s1-20260925-next-gaps-r2`；3 passed。
- 本轮新增定向：`rtk proxy env DRAGON_E2E_PORT=20672 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts --reporter=line --output=/tmp/dragon-next-e2e-s1-mask-switch-20260925`；4 passed。caption commit：`rtk proxy env DRAGON_E2E_PORT=20673 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/caption-context.spec.ts --grep 'caption TXT commit' --reporter=line --output=/tmp/dragon-next-e2e-s1-caption-commit-20260925`；1 passed。训练启动失败与监控核对链接：`rtk proxy env DRAGON_E2E_PORT=20679 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/training-devices.spec.ts --grep 'training start failures' --reporter=line --output=/tmp/dragon-next-e2e-s1-training-start-20260925-monitor-link`；1 passed。
- 上轮前端检查：`rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check`；当时 TypeScript 与 Vitest 49 文件 / 221 用例通过。其并行运行里曾有 1 项数据集预览 error 状态断言失败，隔离复跑及当时完整复跑通过；本轮又复现该状态断言时序问题，修正与当前结果见本记录前文。
- 隔离生产模式构建：`rtk proxy timeout 180 pnpm --dir web/frontend-next exec vite build --outDir /tmp/dragon-next-s1-build-20260925-r1 --emptyOutDir`；通过，输出未发布。
- dirty 数据集入口定向：`rtk proxy env DRAGON_E2E_PORT=20624 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-draft.spec.ts --grep 'dataset dirty save protects image workspace entry' --output=/tmp/dragon-next-e2e-20260925-s1-targeted`

下一步：最新全量与门禁见文末 r25 更新；对账八页逐项状态矩阵，安排真实读屏器及剩余视口/主题/浏览器缩放人工核验。batch-start 部分入队仍属 Next 当前范围外；蒙版 apply 没有逐子集部分结果契约，持久态需在隔离 HTTP/临时根验证。真实 launcher、服务写入、provider、训练和发布静态包继续保持 `NOT RUN`，完成前 S1 不签收。

### S1 续审更新（2026-09-25，r18）

本轮仅使用 mock API/WS，未连接真实服务、训练 launcher、provider 或用户数据根。新增/修复如下：

- 队列 retry 遇到服务端已接受但响应丢失时，不自动重发；刷新取得服务器快照后从 error 收敛为 queued，定向回归通过。
- 历史批量 archive/unarchive/delete 的确认、目标 ID、pending 与“不删除运行目录和权重”边界已有 mock E2E。复现 delete 已在服务端提交但响应丢失时 UI 留下旧错误和失效勾选项；`HistoryPage.tsx` 现于显式刷新成功后清除批量动作错误、仅保留仍在已加载快照中的选择，并显示“当前列表已核对”。`history-actions.spec.ts` 2/2 通过。
- 数据集导入后选中服务端返回的 preset，再通过浏览器下载导出；文件名和 TOML 字节均与输入一致，`dataset-import-export.spec.ts` 1/1 通过。此为浏览器/mock 下载验证，不是磁盘持久化验收；其他 preset CRUD 与冲突/断网矩阵仍有缺口。
- TypeScript 通过；串行 Vitest **50 文件 / 225 用例通过**；`rtk git diff --check` 通过。
- 完整 mock Playwright r18：**189 项，186 passed、2 skipped、1 failed**，输出在 `/tmp/dragon-next-e2e-s1-20260925-r18/`。唯一失败是 `training-library-names.spec.ts` 1440 用例等待组重命名按钮 45 秒；同一 spec 的 1440/390 单 worker 定向复跑 **2/2 passed**（`/tmp/dragon-next-e2e-s1-library-names-r1/`）。目前没有确认产品缺陷，但 r18 全量仍不能记为 PASS；该超时未复现，仍保留为验证波动。

S1 仍为 **NOT RUN / 未签收**。剩余项包括：真实读屏器朗读与 live-region 顺序；尚未覆盖的代表主题/真实浏览器缩放人工复核；数据集 preset 其他 CRUD 与失败矩阵；队列 item stop/remove 的 404/5xx/断网结果核对；模型排序/长路径；损坏或不完整 checkpoint 场景。训练 launcher/真实监控状态、真实写入继续不执行，不能由 mock 推断通过。V-07/V-08 与本轮修复均只存在未发布源码，未替换 `web/static/dragon-next/`。

### S1 续审更新（2026-09-25，r19）

本次完整运行使用当前工作树，分支 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`；工作树仍有大量既存修改，运行期间新增的训练配置库详细管理改动也纳入当前测试。没有清理、回滚或提交其他人的修改。

- 最新完整 mock API/WS Playwright：**190 项，188 passed、2 skipped、0 failed**，单 worker 13.1 分钟；输出在 `/tmp/dragon-next-e2e-s1-20260925-r19/`。2 项 skip 仍是折叠分组拖动场景。该运行覆盖本轮新增 history、queue、dataset 用例及后来出现的训练库详细管理测试。
- 当前训练库相关两份 E2E 定向 **6/6 passed**（`/tmp/dragon-next-e2e-s1-training-library-r2/`）；TypeScript 通过；串行 Vitest **50 文件 / 226 用例通过**；`rtk git diff --check` 通过。
- 文档完整性门禁在 r18 文档编辑后 **8/8 passed**。r18 曾有一条训练库名称用例超时，后续单测及 r19 当前树全量均通过；保留 r18 记录，不把它改写成全绿。

S1 仍为 **NOT RUN / 未签收**。上述均为 mock/API/WS 证据；未完成真实读屏器朗读与 live-region 顺序、代表主题及真实浏览器缩放人工核验。数据集其他 CRUD/失败矩阵、队列 item stop/remove 失败和结果未知、模型排序/长路径、损坏 checkpoint 仍待覆盖。训练 launcher、真实服务写入、provider 调用、训练和发布静态包继续 `NOT RUN`；mock 成功不能替代它们。

### S1 续审更新（2026-09-25，r20）

本轮基于当前工作树继续 mock-only 核验；分支 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`，保留原有大量未提交修改，没有清理、回滚或提交。

- `queue-monitor-mutations.spec.ts` 新增运行中 stop 的响应丢失场景：服务端状态先变为 canceled 再 abort DELETE；客户端保留 running 旧快照、提示结果待确认、不自动重试，显式刷新后收敛为 canceled 并清除错误。与 damaged checkpoint 浏览器链路一起，queue/history 两份定向 spec **14/14 passed**。
- `s3-queue-monitor-history.spec.ts` 新增不完整 optimizer checkpoint：页面呈现 `unavailable_reason`，确认按钮 disabled，强制尝试点击也没有发出续训 POST。使用完整 mock artifacts 响应，不访问 history 文件。
- 新增 `dataset-crud-mutations.spec.ts`：mock 内存库串起新建草稿、保存、另存、重命名和删除；同步双击另存确认只产生一次请求；删除返回 409 时保留既存预设并显示错误，用户显式重试成功。该复测未确认真实产品缺陷。
- 新增 `model-library-order-path.spec.ts`：验证长模型路径完整进入控件、上移/下移边界 no-op、DOM 顺序与 PUT `groups[].item_ids` 一致，路径和顺序在响应后保持。mock-only，无磁盘写入。
- 四份相关 E2E 合并定向 **16/16 passed**；完整 mock API/WS Playwright r20 **195 项：193 passed、2 skipped、0 failed**，单 worker 13.1 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r20/`。2 项 skip 是既有 dataset-drag 折叠组拖动场景。suite 中预期 lazy-import 故障注入有 Vite console error，由路由错误边界接住，不是未处理崩溃。
- TypeScript 通过；串行 Vitest **50 文件 / 226 用例通过**（122.04 秒）；`rtk git diff --check` 通过；文档完整性测试在本轮更新后 **8/8 passed**。

S1 仍为 **NOT RUN / 未签收**：真实读屏器/live-region 朗读顺序、剩余主题及真实浏览器缩放人工核验、逐页完整状态矩阵、队列其他 stop/remove/retry 失败及未知结果、数据集另存/重命名 offline/409/unknown 矩阵、caption job cancel/rerun 失败矩阵、真实 launcher/服务联通仍未关闭。真实写入、训练、provider 调用、构建发布静态包继续 `NOT RUN`；mock 证据不替代人工与隔离 HTTP 验收。

### S1 续审更新（2026-09-25，r21）

- `queue-monitor-mutations.spec.ts` 新增 completed remove 响应丢失场景：服务端先移除项目再 abort DELETE；UI 保留旧 done 卡片并显示结果待确认，显式刷新后从快照移除、清除错误，且请求仅一次。定向 **1/1 passed**（`/tmp/dragon-next-e2e-s1-queue-remove-unknown-r1/`）。
- 当前树完整 mock API/WS Playwright r21 为 **196 项：194 passed、2 skipped、0 failed**，单 worker 13.1 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r21/`。2 项 skip 为既有 dataset-drag 折叠组拖动场景；未连接真实 API/WS。
- 本轮增加的是测试与审计记录，无产品逻辑改动；真实写入、训练、provider 和静态包发布仍为 `NOT RUN`。TypeScript 在新增队列用例后复跑通过；Vitest **50 文件 / 226 用例**结果来自 r20，期间未修改 Vitest 覆盖的源码/单测；`rtk git diff --check` 和文档完整性 **8/8 passed**。

S1 仍为 **NOT RUN / 未签收**。队列 retry/stop 的 404/5xx/断网状态、数据集另存/重命名失败矩阵、caption job cancel/rerun 失败矩阵、逐页状态矩阵、真实读屏器与浏览器缩放人工检查、launcher/服务联通仍未完成；mock 验证不能替代人工或隔离 HTTP 证据。

### S1 续审更新（2026-09-25，r22）

- `caption-context.spec.ts` 新增运行中打标任务取消成功路径：确认文案明确；仅发送一次 `POST /api/captioning/jobs/caption-1/cancel`；pending 时取消、重跑和项目选择锁定；返回 `canceled` 后状态正确并恢复终态控件。定向 **1/1 passed**；整份 spec **16/16 passed**。
- 当前树完整 mock API/WS Playwright r22：**197 项，195 passed、2 skipped、0 failed**，单 worker 13.2 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r22/`。2 项 skip 为既有折叠分组拖动场景；未连接真实 API/WS。TypeScript 通过；本轮只新增 E2E，Vitest 沿用 r20 的 50 文件 / 226 用例结果。
- 本次没有修改产品逻辑。`rtk git diff --check` 通过，文档完整性 **8/8 passed**。mock 证据不覆盖真实任务中断/provider 行为；取消 409/响应未知、rerun 500/断网结果核对仍待补。

S1 仍为 **NOT RUN / 未签收**：队列 retry/stop 其他错误及结果未知、数据集另存/重命名失败矩阵、caption cancel 409/结果未知与 rerun 失败矩阵、逐页状态矩阵、真实读屏器/live-region 与浏览器缩放人工检查、真实 launcher/服务联通均未关闭。真实写入、训练、provider 调用和发布静态包继续 `NOT RUN`；mock 验证不能替代人工或隔离 HTTP 验收。

### S1 续审更新（2026-09-25，r23）

- `dataset-crud-mutations.spec.ts` 增加另存并发 409 路径：失败时保留当前文件及 dirty 表单，显示错误，另开对话框后可用同一草稿成功另存。`queue-monitor-mutations.spec.ts` 增加 retry 503 路径，断言请求仅一次、pending 时按钮锁定、错误可见、旧 error 卡片保留且失败后可重试。两份相关 E2E **14/14 passed**（`/tmp/dragon-next-e2e-s1-failure-matrix-r1/`），没有复现产品缺陷。
- 当前树完整 mock API/WS Playwright r23：**198 项，196 passed、2 skipped、0 failed**，单 worker 13.1 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r23/`。2 项 skip 为既有折叠分组拖动；TypeScript 通过，Vitest 沿用 r20 的 50 文件 / 226 用例结果；文档完整性 8/8、`rtk git diff --check` 通过。
- 这些测试只验证前端 mock 状态语义，不证明真实服务端持久态。仍需覆盖 retry 404、running stop 的 404/5xx、数据集 rename/另存 offline 或响应未知、caption cancel 失败语义。

S1 仍为 **NOT RUN / 未签收**：上述剩余服务端错误/未知状态、逐页完整状态矩阵、真实读屏器/live-region 与浏览器缩放人工检查、真实 launcher/服务联通均未关闭；真实写入、训练、provider 调用和发布静态包继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r24）

- `dataset-crud-mutations.spec.ts` 将 rename 旧预设删除改为返回 409，验证新旧文件都保留、页面选中新文件、已保存表单值同步，并显示“新预设已保存，但旧预设删除失败”。`caption-context.spec.ts` 新增 cancel 409：错误可见同时刷新到服务端 `done` 快照，取消禁用、重跑恢复。三份 caption/dataset/queue 定向 **31/31 passed**（`/tmp/dragon-next-e2e-s1-mutation-matrix-r1/`）。
- 当前树完整 mock API/WS Playwright r24：**199 项，197 passed、2 skipped、0 failed**，单 worker 13.2 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r24/`；2 项 skip 为既有折叠分组拖动。TypeScript 通过；Vitest 沿用 r20 的 50 文件 / 226 用例结果；文档完整性 **8/8 passed**、`rtk git diff --check` 通过。
- 仅增加/调整 mock E2E 和审计记录，无产品逻辑改动；真实后端/provider/launcher、真实写入与发布仍未执行。

S1 仍为 **NOT RUN / 未签收**：队列 retry 404 与 running stop 404/5xx、数据集 rename 新文件 POST 409/断网结果未知、caption cancel 网络结果未知与 rerun 5xx/断网、完整状态矩阵、真实读屏器/live-region 和浏览器缩放人工核验、launcher/服务联通仍待完成。

### S1 续审更新（2026-09-25，r25）

- 队列新增 retry 404、running stop 404/503 状态恢复用例；失败期间保留旧快照、释放 pending 锁且不自动重试，显式刷新后按服务端快照收敛。数据集新增 rename 首次 save-as 409（旧文件与草稿保留、未发 DELETE）、rename 响应未知（服务端已有新文件但旧文件保留）及另存离线后保留草稿/显式重试。caption 新增 rerun 500、rerun/cancel 响应未知；复现 rerun 已被 mock 服务端接受但响应丢失时仍能再次提交的风险，现于 `CaptionReview.tsx` 将未知结果标记为待核对、锁定操作与审阅控件，用户显式刷新任务快照后才恢复操作。未调用 provider 或真实服务。
- caption 未知结果与成功取消复核 **3/3 passed**（`/tmp/dragon-next-e2e-s1-20260925-r25-caption-unknown-final/`）。当前树完整 mock API/WS Playwright **208 项，206 passed、2 skipped、0 failed**，单 worker 13.5 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r25/`；2 项 skip 仍为 `dataset-drag.spec.ts` 的折叠分组拖动。
- `web-next-check` 通过：TypeScript 与串行 Vitest **50 文件 / 226 用例**；本轮已执行的 `rtk git diff --check` 通过。工作树 `dev` / HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`，分支 ahead 13 且有大量既存修改；未清理、回滚、暂存或提交。静态包未替换。

S1 仍为 **NOT RUN / 未签收**：八页完整状态矩阵尚未逐项对账；真实读屏器/live-region 朗读顺序、剩余视口/主题/真实浏览器缩放人工核验仍缺；launcher/真实服务核对与隔离持久态不由 mock 证明。真实写入、provider、训练和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r26）

- 训练启动 unknown-state 定向用例发现网络断开被 `apiRequest` 包装为 `ApiError(status=0)`，但启动弹窗只将 `TypeError` 和 HTTP 5xx 识别为结果未知，因而缺少“启动结果可能未知”提示。修复 `TrainingLaunchDialog.tsx` 的判定，将 status 0 纳入；断网后继续锁住当前提交并给出监控核对入口，不允许同一弹窗盲目重发。该路径模拟服务端已接受启动、客户端响应丢失，之后由监控快照核对到运行任务；未连接真实 launcher。
- `training-devices.spec.ts` 定向 **7/7 passed**。完整 mock API/WS Playwright r26 **209 项：207 passed、2 skipped、0 failed**，2 worker、8.0 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r26/`；2 项 skip 是既有折叠分组拖动。E2E 使用隔离 Vite 端口 `21785` 和不可达 API target `127.0.0.1:29999`，API/WS 由测试 mock。
- `web-next-check` 通过：TypeScript 通过，Vitest **50 文件 / 226 用例**。工作树仍为 `dev` / HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`、ahead 13；保留既有未提交修改，未清理、回滚、暂存、提交或替换静态包。文档完整性 **8/8 passed**，`rtk git diff --check` 通过。

S1 仍为 **NOT RUN / 未签收**：八页完整状态矩阵尚未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口与真实浏览器缩放人工核验未完成。真实 launcher、服务写入与隔离持久态仍不能由 mock 证明；真实写入、provider、训练及静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r27）

- `HistoryDetailPage` 主详情读取新增两个 mock E2E：请求挂起时显示并清除 loading status；主 API 503 显示 alert 与“重新读取”，用户显式重试后渲染详情并清除错误。测试适配 React StrictMode 的重复挂载和 query 的一次自动读取重试，不将只读 GET 次数误判为重复写入。未连接真实 history 根。
- `history-overview-assets.spec.ts` 全量 **10/10 passed**（`/tmp/dragon-next-e2e-s1-history-detail-r27-full/`）。当前完整 mock API/WS Playwright r27 **211 项：209 passed、2 skipped、0 failed**，2 worker、8.0 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r27/`；2 项 skip 是既有折叠分组拖动。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**；文档完整性 **8/8 passed**，`rtk git diff --check` 通过。
- 本轮仅新增 mock E2E 与审计记录，没有新增真实 API/WS、launcher、磁盘持久态或读屏器证据。工作树仍保留此前未提交修改，未清理、回滚、暂存、提交或替换静态包。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口与真实浏览器缩放人工核验未完成。真实 launcher、服务写入及隔离持久态不由 mock 证明；provider、训练与发布静态包继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r28）

- 历史详情新增预处理任务空态/只读 E2E：无训练 Loss 时显示“不适用训练 Loss”，无配置快照时给出明确说明，非训练任务不显示检查点续训命令；mock 写入记录为空。新用例定向 **1/1 passed**（`/tmp/dragon-next-e2e-s1-history-empty-r28/`）。
- 当前完整 mock API/WS Playwright r28 **212 项：210 passed、2 skipped、0 failed**，2 worker、8.0 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r28/`；两项 skip 是既有折叠分组拖动。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。测试仍使用隔离 Vite/不可达 API target，不连接真实 API/WS。
- 文档完整性、`rtk git diff --check` 在本段更新后复验；本轮只增加 mock E2E 与记录，未执行真实 history 读写、launcher、provider、训练或静态包发布。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口与真实浏览器缩放人工核验未完成。真实服务、launcher 和隔离持久态仍未由本轮证明；provider、训练及发布静态包继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r29-r31）

- r29 完整 mock API/WS Playwright：**213 项，211 passed、2 skipped、0 failed**；r30：**214 项，212 passed、2 skipped、0 failed**，2 worker、约 8 分钟。2 项 skip 均为 `dataset-drag.spec.ts` 的折叠分组拖动。r29 截图抽查覆盖 390px dark 的训练/设置/监控，以及 1440px light 的历史/监控，未见明显遮挡或横向溢出；并发 ECharts 有容器尺寸为 0 的警告，但监控布局单 worker 定向 8/8 未复现，记为观察项。
- r29-r30 增补的 mock 证据包括：懒加载工作区 loading 使用 polite status；模型库初始 GET 409 保持读取错误且不暴露新建/保存控件；训练启动 pending 时按 Escape 不关闭弹窗、不重复提交；历史详情 loading/503 重试和预处理任务空指标/无配置快照状态。均未连接真实 API、WS 或 launcher。
- 本轮审计在 [DatasetPreviewDialog.tsx](../../web/frontend-next/src/features/dataset-editor/DatasetPreviewDialog.tsx) 确认图片预览 pending 只有 `aria-busy`、没有 status 语义；现增加 `role="status" aria-live="polite"`，并用延迟 mock 请求验证加载态出现、成功后清除。`mask-workspace.spec.ts` **11/11 passed**（`/tmp/dragon-next-e2e-s1-mask-status-r2/`）。另加 dirty 蒙版点击图片工作台“返回数据集”用例：取消离开保留蒙版草稿，确认后才离开，零写入；结果确认现有 router blocker 正常，无需改行为。
- 训练配置主读取错误分支原来没有重试入口，读取失败会遮住编辑器与命令。现增加“重试读取”，仅重新读取失败的训练 context/raw query；pending 锁依据实际 fetching 状态，避免 disabled merged query 的 `isPending` 让恢复按钮永久禁用。新增持续 mock 503、用户显式重试后恢复的 E2E；`training-draft-flow.spec.ts` **4/4 passed**（`/tmp/dragon-next-e2e-s1-training-flow-r4/`）。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。当前工作树完整 mock API/WS Playwright r31：**217 项，215 passed、2 skipped、0 failed**，2 worker、8.1 分钟；输出 `/tmp/dragon-next-e2e-s1-20260926-r31/`。运行使用隔离 Vite 端口 `20702` 和不可达 API target `127.0.0.1:29999`，所有 API/WS 由 mock fixture 接管。全量 mock 通过不构成真实服务或持久态证据。
- S1 仍为 **NOT RUN / 未签收**。待办继续包括八页状态矩阵逐项对账、训练 capabilities 错误恢复、历史样张/权重单路失败及产物目录缺失态、监控状态的 live announcement、数据集导入 dirty guard 与已保存预设进入/返回的浏览器往返、真实屏幕阅读器朗读顺序和剩余视口/主题/真实缩放人工核验。真实服务写入、launcher/训练/provider、隔离持久态和静态包发布仍为 `NOT RUN`；未替换 `web/static/dragon-next/`。

### S1 续审更新（2026-09-26，r32）

- 当前 `dev` / HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27` 工作树含大量既存修改，本轮未清理、暂存或提交。完整 mock API/WS Playwright：**221 项，219 passed、2 skipped、0 failed**，2 workers、约 8.3 分钟；运行产物在 `/tmp/dragon-next-e2e-s1-20260926-r32/`。命令为 `rtk proxy env DRAGON_E2E_PORT=20703 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --output=/tmp/dragon-next-e2e-s1-20260926-r32`。API/WS 由 mock fixture 接管；2 项 skip 是既有折叠分组拖动。
- `rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check` 通过：TypeScript 与串行 Vitest **50 文件 / 226 用例通过**。新增覆盖训练 capabilities 目录 503 的显式恢复、历史样张目录缺失/单路失败保留权重结果、监控任务与连接状态播报、dirty 数据集导入保护及已保存预设的图片工作台往返；均为 mock-only。
- 套件中出现 ECharts 容器零尺寸控制台警告；布局/图表边界用例通过，作为观察项保留。未连接真实 API/WS、launcher、provider 或用户数据根，未执行真实写入/训练，也未替换 `web/static/dragon-next/`。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器朗读/live-region 顺序、剩余视口/主题及真实浏览器缩放人工核验未完成。训练 launcher、真实服务和隔离持久态仍未由 mock 证明；真实写入、训练、provider 和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r33）

- `dataset-crud-mutations.spec.ts` 新增“重命名时旧文件删除已提交、响应丢失”场景：DELETE pending 时操作控件和表单锁定；服务端内存 fixture 删除旧文件后 abort 响应；页面显示结果未确认、选中新预设，重新读取的列表只含新文件，DELETE 仅请求一次。定向 **1/1 passed**，完整该 spec **5/5 passed**，运行产物分别在 `/tmp/dragon-next-e2e-s1-dataset-rename-unknown-delete-r1/` 与 `/tmp/dragon-next-e2e-s1-dataset-crud-r2/`。完整 spec 命令：`rtk proxy env DRAGON_E2E_PORT=20705 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r2`。
- `rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。本轮只新增 mock E2E，没有产品源码行为改动；测试使用内存 fixture 和隔离 Vite，不连接真实 API/WS，不落真实预设。
- r32 的 221 项完整 Playwright 运行早于本轮新增用例，仍是最近一次完整套件结果；本轮以该定向 spec 5/5 和前端门禁证明增量改动，没有把旧全量计数当作当前树的完整计数。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；rename 创建阶段断网、另存慢请求 pending 锁、真实读屏器朗读顺序及剩余主题/视口/真实缩放人工检查未完成；真实 launcher、服务与持久态未由 mock 证明。真实写入、训练、provider 和发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r34）

- `dataset-crud-mutations.spec.ts` 新增重命名创建阶段断网恢复：首次 save-as 请求中断时保留旧选择与 dirty 表单、不发送 DELETE；用户显式重试后创建新文件并删除旧文件，列表和表单收敛到已同步状态。完整 mutation spec **6/6 passed**，运行产物在 `/tmp/dragon-next-e2e-s1-dataset-crud-r3/`。
- 命令：`rtk proxy env DRAGON_E2E_PORT=20706 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r3`。新增用例仅使用 mock 内存预设与隔离 Vite，不改产品源码，不连接真实 API/WS。
- 最终 `rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。
- r32 **221 项、219 passed、2 skipped** 仍是最近一次完整 suite；r33/r34 新增的两条 dataset 场景以定向完整 spec 验证，不把旧全量计数外推到当前树。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵未逐项对账；另存慢请求 pending 锁、长名称/路径、真实读屏器朗读顺序和剩余主题/视口/真实缩放人工检查仍待完成。真实 launcher、服务与持久态不由 mock 证明；真实写入、训练、provider 和发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r35）

- `dataset-crud-mutations.spec.ts` 新增另存慢请求状态：POST 等待期间保存、另存、重命名与表单输入均禁用；响应成功后切换到新预设、表单同步，原预设仍存在且没有 DELETE。完整 mutation spec **7/7 passed**，产物在 `/tmp/dragon-next-e2e-s1-dataset-crud-r4/`。
- 命令：`rtk proxy env DRAGON_E2E_PORT=20707 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r4`。最终 `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。
- 本轮只改 mock E2E；最近一次完整 Playwright 仍为 r32，r33-r35 的 dataset 增量以完整 spec 定向验证。未触碰真实服务或持久数据。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵未逐项对账，数据集长名称/路径仍待验证；真实读屏器朗读顺序、剩余主题/视口/真实缩放人工检查、launcher/服务与持久态尚未完成。

### S1 续审更新（2026-09-26，r36）

- 新增长名称/路径布局 E2E：约 120 字符的预设名和路径在 **390×844** 与 **1440×900** 下均保持详情路径可读、列表路径省略显示，document 无横向溢出。`dataset-crud-mutations.spec.ts` **8/8 passed**，运行产物在 `/tmp/dragon-next-e2e-s1-dataset-crud-r5/`。
- 命令：`rtk proxy env DRAGON_E2E_PORT=20709 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r5`。`web-next-check` 最终通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。
- r32 **221 项、219 passed、2 skipped** 仍是最近一次完整 Playwright；r33-r36 的数据集增量以完整 `dataset-crud-mutations.spec.ts` 定向回归证明，未把 r32 计数外推。全部场景使用 mock fixture/隔离 Vite。

S1 仍为 **NOT RUN / 未签收**：数据集专属 dirty/import、CRUD mutation、结果未知、pending 和长路径证据已补齐本轮指定矩阵项；八页其余状态仍未逐项对账。真实读屏器朗读顺序、剩余主题/视口/真实缩放人工检查、launcher/服务与持久态仍未完成。

### S1 续审更新（2026-09-26，r37）

- 新增 caption 取消 HTTP 500 mock E2E：验证 alert 可见、任务快照仍为 `running`、取消按钮恢复为用户可控重试、重新打标仍禁用，且等待 1.2 秒没有自动重发；显式第二次确认后 mock 任务才变为 `canceled`。`caption-context.spec.ts` **21/21 passed**，产物 `/tmp/dragon-next-e2e-s1-caption-cancel-r37/`。
- 补齐历史检查点续训“队列已接受但 POST 响应丢失”用例：确认 pending 锁、未知结果提示、历史/队列核对入口、队列快照收敛及单次提交均通过；`s3-queue-monitor-history.spec.ts` **4/4 passed**，产物 `/tmp/dragon-next-e2e-s1-history-queue-r3/`。fixture 的接受与队列状态均为内存 mock，不是服务端持久化证据。
- 最终 `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。caption/history spec 分别使用隔离 Vite 端口 `20710`、`20711` 和不可达 API target `127.0.0.1:29999`。r32 **221 项，219 passed、2 skipped** 仍为最近一次完整 Playwright；r33-r37 增量以各完整定向 spec 验证，未外推旧全量计数。没有连接真实 API/WS、launcher、provider 或持久数据。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵仍未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r38）

- 历史详情“产物”页补充样张和权重并行失败/独立恢复的 mock E2E：两路 503 分别可见；样张显式重试后权重错误仍在，权重再独立重试并收敛到空态；各自重试只增加一次 GET，fixture 写入记录为空。完整 `history-overview-assets.spec.ts` **12/12 passed**，产物 `/tmp/dragon-next-e2e-s1-history-assets-r38/`。现有组件行为正确，本轮没有产品逻辑修改。
- 测试使用隔离 Vite 端口 `20712`、不可达 API target `127.0.0.1:29999`，仅由 mock fixture 回应；不证明真实 history/output 目录、API、launcher 或持久化。
- r32 **221 项，219 passed、2 skipped** 仍是最近一次完整 Playwright；r33-r38 的增量仅由对应完整定向 spec 验证，未把旧全量计数外推。

S1 仍为 **NOT RUN / 未签收**：八页其余状态仍未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口/真实缩放人工核验，以及真实 launcher、服务和隔离持久态仍未完成。

### S1 续审更新（2026-09-26，r39）

- 当前工作树完整 mock API/WS Playwright **228 项，226 passed、2 skipped、0 failed**，2 workers、8.4 分钟，输出 `/tmp/dragon-next-e2e-s1-20260926-r39/`。2 个 skip 是 `dataset-drag.spec.ts` 的折叠分组 hover 和折叠 header drop 场景。
- 全量已纳入 r33-r38 数据集、caption 与历史增量用例。运行使用隔离 Vite 端口 `20713` 与不可达 API target `127.0.0.1:29999`，API/WS 由 mock fixture 接管；没有连接真实 API/WS、launcher、provider、用户目录或持久态。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。文档完整性与 `rtk git diff --check` 在本段更新后复验。
- 状态矩阵已依据当前 E2E 更新历史、caption 与模型/设置行；未覆盖项明确保留为 resume-options 读取失败、切换根后 refetch 失败、provider ping 断网，以及真实服务/人工验收边界。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口/真实浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态尚未完成。真实写入、训练、provider 和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r40）

- 新增 provider ping 网络中断 mock E2E：第一次 POST abort 后显示 status 0 的连接中断/结果未确认文案，pending 时按钮禁用、恢复后无自动重试；用户再次确认触发第二次 ping 并成功。`interactions.spec.ts` **10/10 passed**，输出 `/tmp/dragon-next-e2e-s1-caption-ping-r40/`。
- 本次增量使用隔离 Vite 端口 `20714` 与不可达 API target `127.0.0.1:29999`，只通过 Playwright route mock；未访问真实 provider。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。
- 完整 Playwright r39 为 **228 项、226 passed、2 skipped**，早于本次新增用例；r40 仅以完整 `interactions.spec.ts` 定向 10/10 验证该增量，未把 r39 全量计数外推。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口/真实浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态尚未完成。

### S1 续审更新（2026-09-26，r41-r43）

- 设置根切换后的新根读取失败补测：先缓存旧根数据并成功切换，再让新根 dataset GET 连续返回 503；页面显示错误且不泄漏旧数据，用户显式重试后读到新根数据，设置 PUT 仅一次。`s2-isolation.spec.ts` **4/4 passed**，产物 `/tmp/dragon-next-e2e-s1-root-switch-r41/`。
- 历史续训选项查询失败补测：resume-options 503 时显示错误、确认和提交禁用、无自动重试/无续训 POST；显式“重试读取”成功后再解锁确认。`s3-queue-monitor-history.spec.ts` **5/5 passed**，产物 `/tmp/dragon-next-e2e-s1-resume-options-r42/`。
- provider ping 网络中断用例覆盖 status 0 错误、pending 锁、无自动重试与显式重试，`interactions.spec.ts` **10/10 passed**。当前完整 mock API/WS Playwright **231 项，229 passed、2 skipped、0 failed**，2 workers、8.6 分钟，产物 `/tmp/dragon-next-e2e-s1-20260926-r43/`；两个 skip 仍是折叠分组 drag 场景。
- 全量使用隔离 Vite 端口 `20717` 和不可达 API target `127.0.0.1:29999`，API/WS 由 mock fixture 接管。前端门禁 TypeScript 与 Vitest **50 文件 / 226 用例通过**；文档完整性 **8/8**、`rtk git diff --check` 通过。无真实 API/WS、launcher、provider、用户数据或持久化访问。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵及其他 workspace query 的完整失效范围仍待逐项核对；真实读屏器/live-region 朗读顺序、剩余主题/视口/真实浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r44）

- 扩展配置根切换 mock E2E：一次隔离设置变更同步切换 `configs_root`、`history_root`、`queue_root`，随后验证训练配置 groups/presets/merged、数据集、模型、历史和队列缓存均重新读取各自的新根数据；保留新根 dataset GET 503、错误可见及显式重试恢复场景。完整 `s2-isolation.spec.ts` **4/4 passed**，输出 `/tmp/dragon-next-e2e-s1-root-invalidation-r44d/`。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20721 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s2-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-root-invalidation-r44d`。只扩展 mock E2E，无产品逻辑改动；未连接真实 API/WS、launcher、provider 或持久数据。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**；文档完整性 **8/8 passed**，`rtk git diff --check` 通过。r44 只跑了隔离根切换 spec，不将 r43 的 231 项完整套件计数外推到新用例。
- 该测试覆盖的是代表性根相关 query 家族，不等于已穷尽每个 workspace 的所有 query key，也不替代读屏器、主题/缩放人工验收或真实服务验证。

S1 仍为 **NOT RUN / 未签收**：全工作区 query key 与状态矩阵仍未完全盘点；真实读屏器/live-region 顺序、剩余主题/视口/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r45）

- 历史续训增加混合检查点 mock E2E：可用项与缺少 `optimizer.bin` 的不可用项同时存在；切到不可用项后原因可见且确认/提交禁用，切回可用项会清除先前确认，重新确认后只提交该有效路径。`s3-queue-monitor-history.spec.ts` **6/6 passed**，输出 `/tmp/dragon-next-e2e-s1-history-mixed-checkpoint-r45/`。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20722 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s3-queue-monitor-history.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-history-mixed-checkpoint-r45`。使用 mock API/WS；未连接真实 history/output、launcher、provider 或持久数据。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。新增用例先经完整定向 spec 验证；当前树的完整 Playwright 结果见 r46。
- 该结果补足 Next UI 的选择/确认绑定证据，不覆盖后端 checkpoint 文件变体的全部完整性组合，也不替代真实持久态检查。

S1 仍为 **NOT RUN / 未签收**：其他完整性字段组合、全工作区 query key 与状态矩阵仍未完全盘点；真实读屏器/live-region 顺序、剩余主题/视口/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r46）

- 当前完整 mock API/WS Playwright **232 项，230 passed、2 skipped、0 failed**，2 workers、8.6 分钟；输出 `/tmp/dragon-next-e2e-s1-20260926-r46/`。两个 skip 是 `dataset-drag.spec.ts` 的折叠分组 hover/drop 场景。全量包含 r44 根切换扩展与 r45 混合 checkpoint 用例。
- 命令：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20723 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r46`。全程使用隔离 Vite 与 mock API/WS；未连接真实服务、launcher、provider、用户数据或持久态。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**；本节更新后的文档完整性 **8/8 passed**、`rtk git diff --check` 通过。完整 mock 通过不等于真实读屏、视觉矩阵、服务联通或发布验收。

S1 仍为 **NOT RUN / 未签收**：其他完整性字段组合、全工作区 query key 与八页状态矩阵仍未完全盘点；真实读屏器/live-region 顺序、剩余主题/视口/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r47）

- 蒙版工作区新增两个 mock-only 读取恢复场景：列表 GET 503 时画布保持不可编辑，显式“重试加载”后蒙版列表与画布恢复；预设 GET 持续 503 时保留已加载的蒙版页，显式重试后清除错误。preset query 按当前全局策略先自动重试一次，测试让 mock 持续失败直至用户操作；蒙版 GET/预设 GET 均未产生写入。`mask-workspace.spec.ts` **13/13 passed**，输出 `/tmp/dragon-next-e2e-s1-mask-retry-r48/`。
- 初轮夹具按“首个请求失败”注入时，开发模式 StrictMode 取消了首个 GET，后续请求按夹具预期成功；调整为错误持续至显式放行，避免把挂载请求取消误判为产品行为。本轮未改产品源码。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20725 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-mask-retry-r48`。隔离 Vite 与 mock API/WS；无真实 API、launcher、provider、用户目录或持久数据访问。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。r46 **232 项、230 passed、2 skipped** 仍是最近一次完整 Playwright；本轮仅验证完整 `mask-workspace.spec.ts`，不把增量计入全套通过数。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵及完整 query key 尚未逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r71、r74、r76-r77）

- 扩展 `s2-isolation.spec.ts` history 根切换用例，核验同一 task ID 的 resume-options 在 `history` 与 `external-history` 返回不同 checkpoint；新根 GET 挂起期间旧根 checkpoint 不再是选中值，响应后选择值来自新根。没有发送续训 POST；fixture mutation 记录为空。
- 完整 `s2-isolation.spec.ts` **9/9 passed**，输出 `/tmp/dragon-next-e2e-s1-history-resume-r74/`。命令：`rtk proxy env DRAGON_E2E_PORT=20960 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s2-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-history-resume-r74`。仅隔离 Vite 与 route mocks；不证明真实 history 根持久态。
- 新增 `history-root-log-isolation.spec.ts`，在两个 history root 对同一 task 验证 log metadata (`limit=1`) 与选中分页 (`offset=400&limit=400`)；新根 metadata pending 时旧总行数和日志行不显示，响应后页面收敛到新根。定向 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-history-logs-r76/`；fixture 无写入/未处理请求。
- 扩展 `captioning-root-isolation.spec.ts` 到 61 张 fixture 图片：切换到新根后验证 `offset=0`，再触发 `offset=60&limit=60`；新页 pending 时旧页图片撤下，放行后显示新根第 61 张图片。完整 spec **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-caption-offset-r77/`；没有创建打标 job 或调用 provider。
- TypeScript 通过；单 worker Vitest **53 文件 / 230 用例 passed**。一次并行 `web-next-check` 的 Vitest 阶段为 53 文件 / 226 passed、4 failed（含 3 个 5s timeout 和 1 个配置加载等待失败）；单 worker 全量重跑 53/230 passed，未发现本轮改动引起的失败。文档完整性 **8/8 passed**。五份未跟踪目标文件的 `git diff --no-index --check /dev/null <file>` 均无 whitespace 报告，命令返回 1 仅表示文件与 `/dev/null` 存在差异。
- 最近一次完整 mock API/WS Playwright 仍为 r59：**241 项，239 passed、2 skipped、0 failed**；本次只跑完整隔离 spec 与日志定向用例，不把增量计入 r59。没有连接真实 API/WS、launcher、provider 或用户数据。

S1 仍为 **NOT RUN / 未签收**：history collections/log search 与其他分页组合、captioning profiles/jobs/detail/logs/prompts/assets/dictionary query、完整八页状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r49-r50）

- 队列单项 move 新增 HTTP 409 与 503 mock E2E：核对 POST 目标 `queue-a`、body `{ direction: "down" }`、原顺序保留、pending 后按钮解锁、显示错误且等待后仍仅一次请求。完整 `queue-monitor-mutations.spec.ts` **18/18 passed**，输出 `/tmp/dragon-next-e2e-s1-queue-move-http-r49/`。
- 历史 archive 新增 409 与 503 mock E2E：确认仅归档所选 task；pending 时锁按钮，失败后任务和选择保留、错误可见、按钮恢复且无自动重试；显式刷新核对后清错误并保留仍存在的选择。完整 `history-actions.spec.ts` **4/4 passed**，输出 `/tmp/dragon-next-e2e-s1-history-archive-http-r50/`。
- 命令均使用隔离 Vite 与不可达 API target `127.0.0.1:29999`，端口分别为 `20726`、`20727`；route fixture 接管 API/WS，无真实队列、history/output、launcher 或持久数据访问。本轮只补 mock E2E，没有发现或修改产品逻辑。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。r46 **232 项、230 passed、2 skipped** 仍为最近一次完整 Playwright；本轮以两个完整定向 spec 验证增量，不把旧全量结果外推。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵及完整 query key 尚未逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。
