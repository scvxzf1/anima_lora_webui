# Guidelines 索引

一句话：这里是用户和维护者最常用的安装、训练、推理操作说明。

状态：稳定
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [指南书.md](指南书.md) | 中文综合指南，覆盖安装、数据集、WebUI、训练、推理和 ComfyUI 部署 |
| [linux-deployment.zh.md](linux-deployment.zh.md) | Linux 部署与启动指南 |
| [git-sync-policy.md](git-sync-policy.md) | 本地 `main` 与线上 `webui/main` 的同步规则 |
| [training.md](training.md) | 训练参考：LoRA 变体、caption shuffle、masked loss、数据集配置 |
| [inference.md](inference.md) | 推理参考：推理命令、DCW、Spectrum、prompt 文件 |
| [difference_between_comfy.md](difference_between_comfy.md) | anima_lora 与 ComfyUI 核心实现差异 |
| [guidebook.md](guidebook.md) | 英文综合指南 |
| [ガイドブック.md](ガイドブック.md) | 日文综合指南 |
| [가이드북.md](가이드북.md) | 韩文综合指南 |

## 维护规则

- 中文主路径优先；翻译文档可滞后，但不能成为唯一事实来源。
- 用户安装、训练、推理入口变更时，优先更新本目录。
- 方法细节仍应链接到 `docs/methods/` 或 `docs/experimental/`，避免把算法正文堆进指南。
- 新增文档后，同步更新本索引和 [../README.md](../README.md)。
