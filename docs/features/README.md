# 功能文档索引

这里放 WebUI / GUI 的独立功能说明，不放算法方法正文。

## 当前文档

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [config-workbench.md](config-workbench.md) | 用户功能说明 | 配置页：预设管理、表单编辑、启动训练与加入队列 |
| [dataset-editor.md](dataset-editor.md) | 用户功能说明 | 数据集页：可复用 dataset 蓝图、分组与预览 |
| [training-queue.md](training-queue.md) | 用户功能说明 | 训练页队列：排队、暂停、失败策略与批量中止 |
| [history-collections.md](history-collections.md) | 用户功能说明 | 历史任务与集合：筛选、归档、批量操作与详情 |
| [preview.md](preview.md) | 用户功能说明 | 训练样张 / 推理预览 / 权重列表 |
| [global-settings.md](global-settings.md) | 用户功能说明 | 输出根、模型默认路径、配置根、界面缩放 |
| [ui-scale.md](ui-scale.md) | 用户功能说明 | UI 缩放：默认比例与分页面独立比例 |
| [frontend-health-scorecard.md](frontend-health-scorecard.md) | 维护用评分入口 | 快速审核当前分支前端健康度的规范评分结构、基线与五轮门禁 |

## 维护规则

- 如果功能是训练方法或推理方法，放到 `docs/methods/` 或 `docs/experimental/`。
- 如果功能是 WebUI / GUI 的独立体验、设置或面板，放到本目录。
- 新增功能文档后，同步更新本索引和 [../README.md](../README.md)。
- 用户向功能文档至少写清：入口、关键配置项、危险项、相关测试。
