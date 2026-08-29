# Captioning 13 界面外部 UI/UX 评审记录

状态：进行中  
日期：2026-08-29  
范围：Dragon UI Captioning 工作台 13 个界面，桌面 `1440x900` 与移动端 `390x844`

## 目标与方法

- 使用 `frontend-uiux-review` 的评审脚本和 rubric，将本地 Playwright 截图发送到用户授权的 OpenAI-compatible 视觉服务。
- 计划使用 `gemini-3.6-flash` 与 `gemini-3.1-pro-preview` 对同一证据独立评审，再由 Codex 根据截图、源码和交互实测裁决。
- 本地证据目录为 `/tmp/captioning-uiux-20260829/`：13 张桌面截图、13 张移动端截图及补充复核图。
- API key 只通过进程环境传入，不写入仓库、命令输出或评审结果。

## 阶段 01：API 调度探针

日期：2026-08-29  
结果：鉴权和模型发现成功，视觉 completion 未成功

### 已验证

- `GET https://www.lmproxy.de5.net/v1/models` 返回 HTTP 200，共 37 个模型。
- 目标模型 `gemini-3.6-flash` 和 `gemini-3.1-pro-preview` 均在模型列表中。
- 未带 key 的同一请求返回 HTTP 401，带 key 后返回 HTTP 200，证明用户提供的凭据已被服务接受。

### 视觉请求结果

1. 输入 `desktop-montage-0.png`，模型 `gemini-3.6-flash`，限制 1200 个输出 token。网关在 134.65 秒后断开连接，脚本记录 `RemoteDisconnected`。
2. 改用单界面 `desktop-files.png`，缩放上限 1200px，限制 900 个输出 token。服务在 79.45 秒后返回 HTTP 502。
3. 两次请求均没有返回可用的评审文本，因此本阶段不对界面结论做任何模型归因。

### 裁决与下一步

- 截图大小不是唯一原因：单界面小图仍返回 502，当前阻塞在外部服务的视觉 completion 链路。
- 下次先用极小视觉请求或 provider 实际视觉测试验证通道，再重试评审脚本；不在未通的链路上批量发送 26 张截图。
- 通道恢复后，按界面或小批次运行双模型评审，保存结构化输出，对每条建议标记“接受 / 降级 / 拒绝”及可验收标准。

## 阶段验收

- 已完成一次带授权凭据的 API 调度尝试和一次缩小证据后的复测。
- 已确认 base URL 必须使用 `/v1`；根路径返回非 JSON 页面，不符合评审脚本协议。
- 本阶段没有修改前端、用户配置、Caption 数据或现有截图。
