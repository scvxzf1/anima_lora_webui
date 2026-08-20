# Anima LoRA 文档导航

这里是 Anima LoRA WebUI 的文档总入口。本文只负责把读者带到正确的当前说明、分区索引或历史证据，不重复维护各分区的完整文件清单。

## 按任务查阅

| 你要做什么 | 入口 |
| --- | --- |
| 安装并启动 WebUI | [根 README](../README.md)、[Linux 部署指南](guidelines/linux-deployment.zh.md) |
| 准备数据并训练 | [训练参考](guidelines/training.md) |
| 推理与组合 DCW、Spectrum | [推理参考](guidelines/inference.md) |
| 理解配置合并和外置路径 | [配置索引](configuration/README.md) |
| 使用 WebUI 独立功能 | [功能索引](features/README.md) |
| 切换 Dragon / classic 界面 | [Dragon UI 指南](features/dragon-ui.md) |
| 审核当前分支前端健康度 | [前端健康度评分卡](features/frontend-health-scorecard.md) |
| 查稳定方法 | [方法索引](methods/README.md) |
| 查可运行实验 | [实验索引](experimental/README.md) |
| 理解模型、路由和优化原理 | [结构索引](structure/README.md) |
| 查实验结论、失败路径和审计 | [研究结论索引](findings/README.md) |
| 评估活跃提案或维护计划 | [提案索引](proposal/README.md)、[施工区索引](superpowers/README.md) |
| 准备贡献或认领待办 | [PR 规范](../CONTRIBUTING.md)、[贡献优先事项](contribution-priorities.md) |
| 查归档材料 | [归档索引](archive-index.md) |

## 分区索引

| 分区 | 内容边界 |
| --- | --- |
| [guidelines/](guidelines/README.md) | 安装、训练、推理和维护操作说明 |
| [configuration/](configuration/README.md) | 配置、路径和环境变量 |
| [methods/](methods/README.md) | 稳定或已接入能力的使用说明 |
| [experimental/](experimental/README.md) | 可运行但仍需验证或调参的能力 |
| [structure/](structure/README.md) | 原理、数学和架构 |
| [features/](features/README.md) | WebUI 独立功能 |
| [findings/](findings/README.md) | 实验结果、失败路径、审计和运行报告 |
| [optimizations/](optimizations/README.md) | compile、kernel、显存和性能优化 |
| [proposal/](proposal/README.md) | 活跃或半活跃提案 |
| [superpowers/](superpowers/README.md) | 当前迭代的 spec、plan 和执行日志 |

每个分区的 `README.md` 负责列出该分区文档；本页只链接分区入口。

## 仓库级参考

| 文档 | 用途 |
| --- | --- |
| [contribution-priorities.md](contribution-priorities.md) | 从 PR 规范中分离出的当前贡献方向和验收设想 |
| [optimization-configs-current.md](optimization-configs-current.md) | 与实时配置面同步的优化配置事实清单 |
| [optimization-roadmap.md](optimization-roadmap.md) | 跨方法优化路线图 |
| [multi_model_support.md](multi_model_support.md) | 多模型支持的架构耦合地图 |
| [separation_plan.md](separation_plan.md) | 训练、推理和文档拆分计划记录 |
| [archive-index.md](archive-index.md) | `_archive/docs/` 历史材料入口 |

## 文档生命周期

- 代码、配置和测试是当前行为的最终事实来源；文档冲突时先核实实现，再修正文档。
- 用户稳定路径放在 `guidelines/`、`methods/`、`configuration/` 和 `features/`。
- 可运行但边界未稳定的能力放在 `experimental/`，并在前 25 行标明状态和限制。
- 实验结果、失败结论和审计证据保留在 `findings/`；它们不因结论过时而删除，但必须与当前说明区分。
- 活跃设计放在 `proposal/`；完成、失效或只服务历史合并的材料通过 [归档索引](archive-index.md) 进入 `_archive/docs/`。
- `superpowers/` 是维护施工区，不是用户主路径；完成后应链接到正式说明、findings 或归档材料。
- 新增、移动或归档 Markdown 时，必须更新本页或对应分区索引，保证从本页可达。
- 根 `README.md` 只保留项目介绍、部署快照和最高频入口；深入内容进入本树。

## 验证

文档结构、链接、锚点、围栏、索引完整性、生命周期标记和部分当前配置事实由以下测试守护：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
git diff --check -- README.md AGENTS.md CONTRIBUTING.md docs _archive/docs
```

外部链接只在任务需要时人工核验；不要为了整理导航改写实验结论或历史运行记录。
