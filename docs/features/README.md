# 功能文档索引

这里放 WebUI / GUI 的独立功能说明，不放算法方法正文。

## 当前文档

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [dragon-ui.md](dragon-ui.md) | 历史 / 已退役 | 旧 Dragon/classic UI 的行为记录；不是当前操作指南 |
| [dragon-ui-functional-map.md](dragon-ui-functional-map.md) | 历史盘点 | 旧 Dragon 八个业务/设置界面的功能、关系图与任务生命周期 |
| [dragon-next-implementation.md](dragon-next-implementation.md) | 实施记录 | React 工作台页面对账、状态保护、测试与部署记录 |
| [dragon-next-uiux-progress.md](dragon-next-uiux-progress.md) | 实施进度 | Next UI/UX 方向的阶段状态、证据与安全边界 |
| [config-workbench.md](config-workbench.md) | 功能说明 | Next React `/next/training` 的表单、预检、启动和队列边界 |
| [preprocess-cache-reuse.md](preprocess-cache-reuse.md) | 用户功能说明 | 预处理缓存复用：共享池、A/B/C 开关与删历史语义 |
| [dataset-editor.md](dataset-editor.md) | 用户功能说明 | 数据集页：可复用 dataset 蓝图、分组与预览 |
| [manual-mask-editor.md](manual-mask-editor.md) | 用户功能说明 | Next 独立手绘蒙版、保存与子集训练接入 |
| [tagging-workbench.md](tagging-workbench.md) | 用户功能说明 | Next React 外部 API 打标、候选审阅与 caption 写回边界 |
| [training-queue.md](training-queue.md) | 用户功能说明 | 训练页队列：排队、暂停、失败策略与批量中止 |
| [history-collections.md](history-collections.md) | 用户功能说明 | 历史任务与集合：筛选、归档、批量操作与详情 |
| [history-logs.md](history-logs.md) | 用户功能说明 | 完整日志、虚拟滚动、页/行跳转与全局搜索 |
| [preview.md](preview.md) | 用户功能说明 | 训练样张 / 推理预览 / 权重列表 |
| [global-settings.md](global-settings.md) | 用户功能说明 | 输出根、配置根、界面缩放 |
| [global-model-configs.md](global-model-configs.md) | 用户功能说明 | Anima / Krea-2 模型配置创建、排序、默认项和选择弹窗 |
| [ui-scale.md](ui-scale.md) | 用户功能说明 | UI 缩放：默认比例与分页面独立比例 |
| [frontend-health-scorecard.md](frontend-health-scorecard.md) | 维护用评分入口 | 前端评分结构与历史基线；具体轮次审计结果保存在 `docs/findings/` |

## 维护规则

- 如果功能是训练方法或推理方法，放到 `docs/methods/` 或 `docs/experimental/`。
- 如果功能是 WebUI / GUI 的独立体验、设置或面板，放到本目录。
- 新增功能文档后，同步更新本索引和 [../README.md](../README.md)。
- 用户向功能文档至少写清：入口、关键配置项、危险项、相关测试。
