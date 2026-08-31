# 外部 API 打标工作台

状态：基座可用
适用界面：Dragon UI

## 入口与联动

直接入口：

```text
http://127.0.0.1:20102/?ui=dragon#page/captioning
```

也可以在 **数据集** 页面选择并保存一个数据集预设，然后点击 **打开打标工作台**。页面会带入当前
`dataset_file`、数据组序号和原始图目录；这段上下文只保存在一次性的 `sessionStorage` 项中，不包含
API Key。

## 基本流程

1. 选择数据集预设、图片组和原始图/训练图目录。
2. 勾选当前页图片并填写打标提示词。
3. 在右上角设置按钮中配置 OpenAI-compatible API 地址、模型、超时、重试和并发数。
4. 发送任务，等待逐张候选 caption 返回。
5. 审阅并修改候选文本，再选择写回当前选中项或全部候选。

外部模型返回结果不会自动写盘。普通文本 caption 使用原子写入；已有 `.json`、`.jsonl` 或共享
`captions.json` 的结构化标注会保守拒绝覆盖。

## 外部 API 配置

当前基座使用 OpenAI-compatible：

```text
GET  <base_url>/models
POST <base_url>/chat/completions
```

视觉请求把数据集图片编码为 `data:<mime>;base64,...`，同时发送系统提示词和当前任务提示词。设置页支持：

- `base_url`、`model`、`system_prompt`
- `timeout_seconds`、`retry_count`、`retry_interval_seconds`
- `concurrency`
- `allow_private_network`

API Key 可通过以下环境变量提供，按从左到右的顺序取首个非空值：

```text
ANIMA_CAPTIONING_API_KEY
ANIMA_TAGGING_API_KEY
TAGGING_API_KEY
```

页面保存的普通设置位于已忽略的 `configs/captioning/settings.toml`；本地密钥位于已忽略且权限尽量
收紧为 `0600` 的 `.anima-captioning-secrets.toml`。GET/PUT 响应只返回 `api_key_configured`，不会回填
密钥明文。环境变量密钥不能从页面删除。

## 安全与当前边界

- 默认拒绝本机、私网、link-local 和保留地址；使用本地模型服务时必须明确启用私网 API。
- DNS 解析结果在 aiohttp connector 层再次检查；外部 API 重定向不会跟随，避免跨主机转发密钥。
- 单张图片最大 64 MiB，单个响应体最大 4 MiB；任务并发上限为 8。
- 任务和候选审阅状态保存在当前 WebUI 进程内，重启后不会恢复，也不进入训练队列。
- 当前图片列表每次读取前 48 张；后端任务上限为 500 张，后续可在此基座上增加分页批选。

## API 与验证

canonical API 前缀为 `/api/captioning`，同时保留 `/api/tagging` alias。主要端点包括设置、连通测试、
任务创建/轮询/取消、候选编辑和写回。

定向测试：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_tagging_service.py \
  tests/test_tagging_routes.py \
  tests/test_dragon_tagging_frontend.py -q
```
