# Dragon Next

React 工作台使用 `/next` 入口，复用 aiohttp API；访问服务根路径 `/` 默认跳转到 Next，旧 Dragon/classic 仍可通过 `/?ui=dragon` 和 `/?ui=classic` 访问。
验收状态与已知边界见 [实施记录](../../docs/features/dragon-next-implementation.md)。
后续审阅改进见 [UI/UX 实施进度](../../docs/features/dragon-next-uiux-progress.md)。

## 开发

验证工具链为 Node 24.13.0、pnpm 11.9.0，运行依赖已锁定：

```bash
pnpm --dir web/frontend-next install --frozen-lockfile
DRAGON_API_TARGET=http://127.0.0.1:20102 pnpm --dir web/frontend-next dev
```

通过 Vite 输出的 URL 访问 `/next/training`。后端另行按仓库指南启动；上述命令不启动训练。
开发代理同时转发 `/api` 和 `/ws`。生产由 Python 服务托管静态文件，不需要 Node 常驻。

没有可用后端时，可启动合成数据只读预览（Node 24 和本机 Chrome）：

```bash
DRAGON_PREVIEW_PORT=20521 node web/frontend-next/scripts/preview-isolated.mjs
```

该入口只绑定 `127.0.0.1`，复用 E2E fixture，所有 API 写请求返回 403，没有真实后端代理。
监控是合成快照，WS 显示断开是预期；不能用它验收真实训练、保存或调度。

## 验证与部署

```bash
python tasks.py web-next-check
python tasks.py web-next-e2e
python tasks.py web-next-build
```

维护执行时加 RTK 前缀并优先使用 `.venv/bin/python`。
E2E 使用独立的 5174 端口和系统 Chrome，所有 API 与训练 WebSocket 均被 mock。
并发验证可通过 `DRAGON_E2E_PORT` 指定空闲端口，并给 Playwright 传入独立 `--output` 目录；不会复用或终止已有服务。
未安装 Chrome 时需先配置 Playwright 的浏览器通道，不能把测试改成对用户后端执行写操作。

构建在临时目录完成，先发布带哈希资源，最后原子替换 `web/static/dragon-next/index.html`。
旧入口副本为 `previous-index.html`，旧 chunks 保留；不会清空正在被访问的资源目录。
生产默认不发布 source map。新构建不依赖外部字体/CDN。
需要在发布前做隔离候选构建时，可设置 `DRAGON_NEXT_DESTINATION=/tmp/dragon-next-candidate`
覆盖发布目标；未设置时仍使用 `web/static/dragon-next`，默认行为不变。
候选包可用 `DRAGON_S5_CANDIDATE=/tmp/dragon-next-candidate node scripts/verify-release-candidate.mjs`
核对入口引用的哈希资源、旧 chunk 保留、`previous-index.html`、原子替换和临时根回退。

发布后访问 Python 服务的 `/next/training`，检查 `/next/history` 等深链刷新。
也可显式指定本机后端运行只读浏览器检查，命令不启动服务且会拦截非只读 API 请求：

```bash
DRAGON_VERIFY_URL=http://127.0.0.1:20102 node web/frontend-next/scripts/verify-production.mjs
```

检查八页、旧入口、页面错误、溢出与首路由 JS gzip 体积。报告和截图默认写入
`/tmp/dragon-next-production-check`，可用 `DRAGON_VERIFY_OUTPUT` 指定其他目录。
生产截图可能包含用户信息，不应提交或发送到外部评审服务。

故障时可直接转到 `/?ui=dragon` 或 `/?ui=classic`，不需要重启后端或停止训练。
`web/static/dragon-next/previous-index.html` 保留上一次构建的入口文件。
需要回退静态版本时，先备份当前 `index.html`，将 `previous-index.html` 复制到同目录临时文件，
再原子替换 `index.html`。这只回退前端，不回滚业务数据；不要直接打开副本的静态 URL，router 仍要求 `/next`。

## 维护边界

- 数据集页每个已保存子集通过「图片工作台」进入 `/next/datasets/workspace/{preview|masks|tagging}`。三个视图共享预设/子集上下文；显式返回与浏览器后退会恢复数据集列表滚动位置。预览与手绘蒙版复用现有实现，打标视图带入对应数据集和子集，同时保留打标任务、结果审阅及接入/提示词/本地资源配置入口。蒙版的未保存离开保护仍覆盖视图切换；旧 `/next/datasets/masks` 路由继续可用。详见 [手动蒙版编辑器](../../docs/features/manual-mask-editor.md) 与 [打标工作台](../../docs/features/tagging-workbench.md)。

- 训练页通过“选择与配置数据集”打开独立弹窗，支持蓝图库搜索、多子集原始图片数量、缩略图与展开预览、目录和重复次数编辑、正则化标记及添加/移除子集。数量来自已保存蓝图的原始图片目录，不等同于训练实际样本量；每个子集预览前 8 张。修改共享蓝图参数须确认保存，影响所有引用该蓝图的配置；“使用此数据集”只更新训练草稿，仍需保存训练配置。关闭或切换时保护未保存参数。阶段调度蓝图的子集结构及更多高级参数在“完整蓝图管理”中编辑。

- 数据集蓝图的每个子集在“高级规则 → 蒙版”中配置 `mask_mode` 和 `mask_dir`，普通与正则化子集可独立设置。外部模式要求目录非空，图片 `001.jpg` 对应 `001_mask.png`；白色参与损失，黑色忽略，缺失文件回退整图。整图训练请显式选择“不使用蒙版”，不要依赖关闭全局 `masked_loss`。切换为其他模式保存时会清除外部目录，并同步旧 `alpha_mask` 字段，避免残留配置改变模式。

- 桌面训练配置页固定在视口内，参数区和配置库独立滚动。配置库展开状态保存在当前浏览器；训练量估算、变更预览和预检结果通过弹窗访问。

- 训练设备在参数搜索栏上方常驻显示。单卡使用逐卡单选，多卡数据并行需要显式切换模式并选择至少两张卡；选择仅记在当前浏览器，不写入训练 TOML。已选设备消失或型号变化时阻止提交，需重新选卡。普通预检、启动确认和入队共用同一 GPU 白名单，确认弹窗只显示摘要。后端既有 Accelerate 环境变量覆盖规则仍然适用。
- “设备与性能”集中展示精度、显存管理、编译和诊断设置；预处理与数据加载移到“输入准备”。流水线并行仅保留实验参数审计，主训练执行尚未接入，不能作为可运行模式启用。手机端配置页整体滚动，避免设备栏挤压参数编辑区域。
- 训练量估算显示已保存配置的样本构成、有效批量和步数。分桶面板通过 `GET /api/config/steps?include_buckets=1` 优先实测 `image_dir` 尺寸；训练目录为空时读取源图，复用预处理选桶逻辑按子集配置预测，并应用最低像素过滤，不生成预处理文件。面板明确区分源图预测和训练目录实测，支持数据集筛选、数量/宽高比排序。该分布不包含重复、抽样、验证划分和克隆，不等同于运行时 dataloader 分桶。每个子集最多扫描 20,000 张、10 秒（在图片之间检查），部分结果会显示未读/未统计数量。图片头缓存按路径、修改时间和大小失效，重新估算可刷新目录统计。弹窗与分桶列表独立滚动并隔离滚动边界，弹窗打开期间锁定页面滚动，关闭后恢复。此功能需要更新后端，旧后端会显示分桶不可用提示。

- `app/` 管理壳、路由和 UI 偏好；`features/` 按领域管理表单、API、状态和样式。
- 辅助工具的生图测试、权重分析和环境检测由 Next 独立页面实现，复用现有 HTTP API，不依赖旧界面 DOM 或状态。
- 旧字段目录与能力规则只按纯模块复用，不挂载旧 DOM 或全局业务状态。
- 配置、样张提示词、候选标注、模型库均保留独立草稿；保存不自动执行训练。
- 所有 mutation 禁用自动重试；连接中断按结果未知提示，必须核对服务器后再重试。
- 模型库沿用 revision 冲突控制。普通配置/TXT 的跨标签原子版本保护仍受后端接口能力限制。
- 辅助生图、权重分析与环境工具保留明确的旧界面链接，不伪装为原生 React 工具。
