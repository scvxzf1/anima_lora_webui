# 配置文档索引

这里放配置根目录、路径解析和配置外置相关说明。

## 当前文档

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [external-configs.md](external-configs.md) | 当前实现说明 | 解释 `ANIMA_CONFIGS_ROOT`、WebUI 全局设置里的 `configs_root`、路径解析优先级和迁移建议 |

## 维护规则

- 配置字段、路径环境变量、WebUI 全局设置变更时，优先更新本目录。
- 历史计划和已完成提案放到 `_archive/docs/proposal/`；本机实施快照放到
  `_archive/docs/configuration/`。本目录只保留当前使用说明。
- 示例路径不要写本机绝对路径，除非明确标注为示例。

历史实施快照已移到
[_archive/docs/configuration/](../../_archive/docs/configuration/README.md)。
