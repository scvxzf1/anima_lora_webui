# AGENTS.md

本文件是给 AI Agent 长期维护本仓库用的根级工作协议。它覆盖整个
`anima_lora` 仓库；子目录如果另有 `AGENTS.md` 或 `CLAUDE.md`，以离目标文件更近
的说明为补充约束。根目录曾经通过 `@CLAUDE.md` 引用维护说明，但当前根级
`CLAUDE.md` 可能不存在，因此不要依赖外部展开，优先以本文件和实时源码为准。

## 总体原则

- 默认用简体中文沟通，代码、命令、配置键、错误信息保持项目原有语言风格。
- 先定位子系统，再读最小必要上下文；优先用 `rg` / `rg --files` 查找。
- 保持改动小而准确。不要顺手重构、重排大文件、格式化无关文件或改写历史文档。
- 保护用户运行数据。不要把 `.venv/`、`models/`、`output/`、`post_image_dataset/`、
  `logs/`、`configs/imported/`、`configs/web-training-history/`、
  `configs/web-training-queue/` 当作普通源码清理或覆盖。
- 不要擅自删除模型、缓存、训练结果、历史任务、队列文件或用户导入配置。
- 不要擅自终止训练、清空队列、批量移动输出、下载大模型或启动长训练。确需执行时，
  先说明影响并取得用户明确同意。
- 遇到用户未提交改动，默认那是用户或其他 Agent 的工作。只在任务必须触碰同一文件时
  谨慎合并，不要 revert。
- 代码事实优先于文档。若本文件、旧说明和源码不一致，先读源码和测试，再更新文档。

## 反上帝代码守则

本节用于防止后续维护继续把复杂逻辑堆回少数大文件。

- 热点文件默认只做 facade、编排、兼容 shim 或小范围修复，不新增大块业务逻辑：
  - `train.py`
  - `inference.py`
  - `library/inference/generation.py`
  - `library/datasets/base.py`
  - `networks/lora_anima/network.py`
  - `networks/lora_anima/config.py`
  - `web/services/training_service.py`
  - `web/services/config/_legacy.py`
  - `web/static/js/features/anima-app/chunks/*`
- 修改热点文件超过约 50 行时，优先拆到现有子模块或新模块；若确实不能拆，最终回复要说明：
  - 为什么必须改热点文件。
  - 为什么不能放到新模块。
  - 后续如何继续瘦身。
  - 跑了哪些定向测试。
- 单个新增 Python 函数建议不超过 100 行；超过时优先拆成 helper、pipeline step 或策略对象。
- 单个新增 Python 类建议不超过 400 行；超过时优先按状态、IO、策略、验证、保存/加载拆分。
- 单个新增 JS 函数建议不超过 80 行；新增 UI 逻辑必须按 feature、store、api、renderer 拆分。
- 单个新增测试文件建议不超过 1200 行；超过时按领域拆成多个测试文件，不要继续加大现有超大测试。
- 已超过 1000 行的源码或测试文件，除搬迁、兼容 shim、删除旧逻辑外，默认不继续承载新业务。
- 重构优先采用“搬家型重构”：先保持行为不变地抽模块，再补测试和清理旧 facade；不要一轮同时改架构和改行为。
- 新增配置、CLI、adapter、WebUI 表单或队列/历史行为时，必须同步考虑文档入口、测试入口和旧兼容面，避免逻辑散落。

## 环境和命令入口

- 项目运行环境是 Python 3.13，依赖管理优先使用 `uv`；安装和模型准备以
  [根 README](README.md) 与 [Linux 部署指南](docs/guidelines/linux-deployment.zh.md) 为准。
- [`tasks.py`](tasks.py) 是命令入口真相；`Makefile` 只是薄转发。实时命令面使用
  `python tasks.py --help` 查看，稳定与实验命令实现分别位于
  [`scripts/tasks/`](scripts/tasks/) 和 [`scripts/experimental_tasks/`](scripts/experimental_tasks/)。
- 维护和验证命令优先使用 `.venv/bin/python`；只有确认无需项目虚拟环境，或 `.venv/`
  不存在时，才回退到系统 `python`。
- 跨平台用户文档可写成 `python tasks.py <command>`，本仓维护执行优先写成
  `.venv/bin/python tasks.py <command>`。用户文档可把 `make <target>` 作为兼容写法，
  但不要作为唯一入口。
- `python tasks.py <command> KEY=value` 支持 Make 风格尾随环境变量，例如：
  `python tasks.py print-config METHOD=lora PRESET=default`。
- 实验命令通常以 `exp-*` 开头，可能变动或删除。修改时同步检查实验任务实现、配置、文档和测试。
- 常用维护入口：
  - WebUI：`.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102`
  - 单元测试：`timeout 60 .venv/bin/python -m pytest tests/<test_file>.py`
  - 全量单测：`timeout 60 .venv/bin/python tasks.py test-unit`
  - 合并配置：`.venv/bin/python tasks.py print-config METHOD=<name> PRESET=<name>`
- 老文档中的命令若不在 `tasks.py --help` 当前列表中，按历史或兼容入口处理，不要直接假定仍可用。

## Repowise 代码库地图

- 本机已建立 repowise 索引，Codex MCP 中当前仓库名为 `repowise_anima_lora`；独立
  WebUI 仓库名为 `repowise_anima_webui`。
- 跨模块排查、架构理解、风险分析、dead-code、调用链或符号定位时，优先用 repowise
  获取概览和候选上下文，再读取实时源码确认。
- repowise 索引不是实时真相，不替代 `git diff`、直接读文件和测试验证；新增/删除/重命名
  文件，或修改 import/export、路由、命令、服务注册、公共接口、跨模块调用链后，建议运行
  `uvx repowise update` 刷新地图。
- `.repowise/` 和 `.mcp.json` 是本机索引/本机路径配置，不要提交。

## Git 推送和回滚

- 默认线上目标由仓库和分支确定，而不是由本机 remote 别名确定：
  `github.com/scvxzf1/anima_lora_webui` 的 `main`。用户说“拉取线上更新”“同步线上 main”
  或“推送更新到线上”时，默认指向该目标。
- 操作前先运行 `git remote -v`，找到 URL 匹配目标仓库的 remote；它在不同 checkout 中可能叫
  `origin`、`webui` 或其他名字。命令和汇报使用实际 remote，不要凭文档假定别名存在。
- 指向个人 fork、私有镜像或 `sorryhyun/anima_lora` 等参考仓的 remote 不是默认发布目标。
  除非用户明确点名，不要向它们 pull、push、reset，也不要把上游参考合入和线上发布混为一谈。
- 推送前至少检查：`git status --short --branch`、`git fetch <target-remote> --prune`、
  `git rev-list --left-right --count HEAD...<target-remote>/main` 和
  `git log --oneline <target-remote>/main..HEAD`，再跑与改动直接相关的测试。
  未跟踪文件默认不随推送发布，除非用户明确要求或本次任务确认需要纳入版本控制。
- 用户要求直接“推送更新到线上”时，默认目标是 `<target-remote> main:main`。若当前身份没有
  写权限，使用个人 fork 分支和 PR，不要因此把个人 fork 改称线上主仓。完成后汇报仓库、remote、
  分支、最新提交 hash，以及是否还有未提交或未跟踪改动。
- 用户说“回滚”时，先分清是哪一种：
  - 本地工作区回退：丢弃未提交改动。只有用户明确要求时才做，执行前说明会丢失哪些内容。
  - 本地提交回退：撤销本地一个或多个提交。共享分支默认优先 `git revert`，不要默认改写历史。
  - 线上回退：撤回目标仓库分支上的已发布提交。必须先确认仓库、remote、分支和目标提交。
- 需要以线上仓库为准同步本地时，先 fetch 并比较 `HEAD` 和目标分支，再决定是否 reset。
  不要在没核对差异前直接做破坏性操作。
- `git reset --hard`、`git checkout -- <path>`、`git clean -fd`、`git push --force`、
  `git push --force-with-lease` 都视为高风险操作。除非用户已经明确要求，或已经明确给出
  目标提交/分支并接受影响，否则不要执行。
- 如果确实要改写线上历史，优先使用 `--force-with-lease`，并在执行前说明会覆盖哪个仓库和
  分支、抹掉哪些提交、是否影响其他协作者。
- 可以使用环境变量中的凭据或本机已配置的 SSH key 推送，但不要把 PAT、cookie、私钥或
  带密钥的远程 URL 写进仓库文件、文档、日志样例或长期说明。
- 面向人的精简操作说明见 [Git 同步规则](docs/guidelines/git-sync-policy.md)。

## 项目地图

- `tasks.py`：所有稳定命令注册表。
- `train.py`：`AnimaTrainer` 主训练入口。
- `inference.py`：独立推理入口。
- `anima_lora/`：可安装包门面，给嵌入式调用暴露精选 API。
- `library/`：训练、推理、配置、数据、runtime、模型、captioning、vision 等核心逻辑。
- `library/anima/`：Anima DiT、权重加载、token/text strategy 和模型配置。
- `library/config/`：TOML 读取、合并、normalize、schema 校验。
- `library/training/`：训练 bootstrap、loop、optimizer、scheduler、checkpoint、loss 等。
- `library/inference/`：generation、sampling、adapter 加载、DirectEdit、DCW、输出处理。
- `library/preprocess/`：预处理和缓存编排。
- `networks/`：adapter/network 实现。修改这里前读 `networks/CLAUDE.md`。
- `scripts/tasks/`：稳定命令实现。
- `scripts/experimental_tasks/`：实验命令实现。
- `web/`：aiohttp WebUI 后端和静态前端。
- `preprocess/`：CLI 预处理脚本，底层编排通常在 `library/preprocess/`。
- `custom_nodes/`：ComfyUI 节点；发布副本通过 `_vendor/` 同步。
- `configs/`：base、presets、methods、gui-methods、datasets、Web 设置、历史和队列。
- `tests/`：pytest 测试，按文件名定位覆盖面。
- `bench/`：方法、性能、显存、推理和实验验证。
- `docs/`：文档主树，入口是 `docs/README.md`。
- `_archive/docs/`：历史或缺失上下文资料，不当作当前实现说明。

## 先分类再动手

把需求归入一个主子系统，跨子系统时按依赖顺序处理。

- WebUI：页面、表单、API、训练队列、历史、预览、全局设置、权重分析。
- 配置/数据/runtime：TOML 合并、数据集蓝图、缓存目录、路径解析、运行时配置。
- 训练：bootstrap、dataset、forward、loss、optimizer、resume、checkpoint、progress。
- Adapter/network：LoRA family、LoHa、LoKr、VeRA、GLoRA、Hydra、FeRA、ReFT、
  Chimera、IP-Adapter、EasyControl、BYG、Turbo、SPD、Soft Tokens。
- 推理：generation、sampling、adapter 加载、Spectrum、DCW、SMC-CFG、DirectEdit。
- 预处理：resize、VAE cache、TE cache、PE cache、pooled text cache、caption index、mask。
- Anima Tagger / captioning：tag taxonomy、训练缓存、阈值、DirectEdit source caption。
- Daemon / 队列：本地训练守护进程、WebUI 队列、runtime config、日志追踪。
- Custom nodes：ComfyUI hydralora loader、trainer、tagger、directedit、blockcompile。
- Bench/docs/tests：实验声明、文档维护、验证脚本、测试覆盖。

## 配置和数据约定

### 配置目录外置

配置根目录、训练历史和队列支持外置。优先级、路径展开、安全边界和迁移方法以
[外置配置目录说明](docs/configuration/external-configs.md)及
[`library/env.py`](library/env.py) 为准。

维护时必须保留以下边界：相对路径锚定项目根目录；允许绝对路径、`$HOME` 和 `~`；拒绝
包含 `..` 的路径；未配置覆盖项时继续回退到仓库内 `configs/`。本机路径覆盖写入已忽略的
`.anima-webui-settings.toml`，不要把本机绝对路径写进受版本控制的默认配置。

### 训练配置合并链

```text
configs/base.toml
  -> configs/presets.toml[<preset>]
  -> configs/<methods_subdir>/<method_or_variant>.toml
  -> CLI args
```

- 默认 `methods_subdir="methods"`；WebUI 友好变体使用历史命名目录 `configs/gui-methods/`。
- [`configs/base.toml`](configs/base.toml) 包含共享基础路径、optimizer、compile flag 和默认数据集蓝图。
- [`configs/presets.toml`](configs/presets.toml) 放硬件/采样 profile，不要把硬件 profile 复制进方法配置；实时预设以该文件为准。
- [`configs/methods/`](configs/methods/) 放算法 family 配置。
- [`configs/gui-methods/`](configs/gui-methods/) 放自包含用户变体，不使用注释切换块；实时变体以目录内容为准。
- [`configs/datasets/`](configs/datasets/) 放可复用数据集蓝图。
- `configs/imported/` 是 WebUI 导入或用户配置，默认视为用户数据。
- `configs/sample-prompts/` 放按配置分叉保存的 sample prompts。
- `configs/web-ui-settings.toml` 保存 WebUI 全局设置和模型路径；不要把本机绝对路径写进默认值或文档。

默认数据和缓存：

- 源图片通常在 `image_dataset/`，同名 `.txt` 是 caption master。
- 常见产物：`post_image_dataset/resized/`、`post_image_dataset/lora/`、
  `post_image_dataset/masks/`、`output/ckpt/`、`output/tests/`、`output/runs/`。
- subset 可以设置 `cache_dir`，让 VAE/text/PE sidecar 写入专用缓存目录。
- 重要 sidecar：
  - `{stem}_{WxH}_anima.npz`：VAE latent cache。
  - `{stem}_anima_te.safetensors`：text encoder cache。
  - `{stem}_anima_pe.safetensors`：PE-Core vision feature cache。

## 方法和能力入口

不要在总协议里维护会漂移的方法清单。开始修改某个方法前，按成熟度和任务类型查阅：

- 稳定能力：[方法索引](docs/methods/README.md)。
- 可运行实验：[实验索引](docs/experimental/README.md)。
- 原理与架构：[结构索引](docs/structure/README.md)。
- 实验结论与失败路径：[研究结论索引](docs/findings/README.md)。
- 当前命令入口：[`tasks.py`](tasks.py) 与 `python tasks.py --help`。

文档只用于定位和理解；最终以实时源码、配置和测试为准。

## 不可破坏的不变量

### Text Encoder Padding

Anima 预训练模型需要 max-padded text encoder outputs。padding 位置会作为
cross-attention softmax 的 attention sinks。

- 不要按真实文本长度裁剪 text encoder 输出。
- 不要通过 `crossattn_seqlens` mask 掉 padding。
- tokenizer 或 padding 行为变化后，需要重建磁盘 `.npz` / `.safetensors` 缓存。
- 相关区域：`library/anima/strategy.py`、`library/anima/text_strategies.py`、
  `library/preprocess/text.py`、`library/inference/text.py`。

### Constant Token Buckets 和 Native Flatten

`library/datasets/buckets.py::CONSTANT_TOKEN_BUCKETS` 当前是 24 个 `(W, H)` 分辨率，
分成 4032 和 4200 两个 token-count family。每个 bucket 精确填满自己的 token count，
没有 intra-bucket padding。

- native shapes 是当前唯一模式；不要恢复旧的 pad-to-static 4096 路径。
- `compile_blocks()` 会开启 native-shape flatten，让图按 token count 复用。
- 改 bucket 表、token count、compile flatten、sample prompt 分辨率参与预算时，必须补
  shape/invariant 测试。
- DCW aspect bucket 的顺序会影响已发布 fusion-head checkpoint，不要随意重排。
- 相关测试：`tests/test_constant_token_buckets.py`、`tests/test_native_flatten.py`、
  `tests/test_runtime_harness_cli.py`。

### Lazy Model Loading

训练为了避免 OOM，加载顺序必须保持：

```text
text encoder -> cache -> free
VAE -> cache -> free
DiT -> attach network -> training loop
```

不要把 DiT 提前加载到预处理阶段。WebUI、task runner 和 daemon 启动训练时也要保持
这个顺序。

### Compile After Apply

`torch.compile` 必须 trace adapter monkey-patched forward，所以 `compile_blocks()` 必须在
`network.apply_to` 和 `load_weights` 后执行。复用 `library/runtime/harness.py::build_anima`
或 `compile_blocks_for_training()`，不要在 bench、scripts、preprocess 中手写易错顺序。

### DiT Latent Shape

DiT forward 使用 5D latent：`(B, C, T=1, H, W)`，单例时间轴是 dim 2。

- VAE、cache、训练 inner loop、很多 spectral helper 使用 4D `(B, C, H, W)`。
- 进入 DiT 前显式 `unsqueeze(2)`，离开 DiT 后显式 `squeeze(2)`。
- 不要用裸 `squeeze()` 或 `squeeze(0)` 处理这个边界。

### LoRA Family 三轴路由

LoRA family 路由由三轴配置表达，不要恢复旧 metadata fallback：

- `use_moe_style`: `False` / `"shared_A"` / `"independent_A"`
- `route_per_layer`: `True` / `False`
- `router_source`: `"none"` / `"input"` / `"sigma"` / `"fei"` / `"crossattn_emb"`

关键位置：

- `networks/lora_anima/config.py::LoRANetworkCfg.from_kwargs`
- `networks/__init__.py::resolve_network_spec`
- `networks/lora_anima/network.py`
- `networks/lora_modules/*`

旧 metadata 如 `ss_use_hydra`、`ss_use_fei_router` 不再加载；新 stamp 是
`ss_use_moe_style`、`ss_route_per_layer`、`ss_router_source`。

### GlobalRouter / FEI

当 `route_per_layer=False` 且 `router_source="fei"` 时：

- `network.set_fei(z_t)` 每步计算一次 FEI 和 router。
- routing weights 通过引用写入每个 routing-aware module。
- 训练 loop 和推理 denoising loop 都需要在当前 step 前设置 FEI。

相关位置：`library/runtime/fei.py`、`library/training/router_conditioning.py`、
`library/inference/generation.py`、`networks/lora_anima/network.py`。

### Attention Layout 和 Fused Projection

- `networks/attention_dispatch.py::dispatch_attention()` 是 attention backend layout 转换入口。
- SDPA/sageattn 常见 BHLD；xformers/flash-attn 常见 BLHD。新增 call site 必须明确布局。
- runtime fused `qkv_proj` / `kv_proj` 与 on-disk split `q/k/v_proj` 的唯一真相源是
  `networks/attn_fuse.py`。保存和加载都依赖它。

### Timestep Masking

T-LoRA mask 是共享 buffer，每个 denoising step 更新一次。

- 不要把 `t` 穿透到每个 LoRA forward。
- 新 timestep-aware variant 应复用 buffer pattern。
- `networks/lora_anima/factory.py` 和 `networks/lora_anima/network.py` 是设置/清理 mask 的主要位置。

### Merge 边界

`python tasks.py merge` 只适合可折叠进 DiT Linear 权重的 adapter。

- LoRA / OrthoLoRA / DoRA / T-LoRA 通常可折叠。
- ReFT / Hydra MoE / postfix / IP-Adapter / EasyControl / BYG / Turbo / Soft Tokens 通常不能完整折叠。
- 新方法必须更新 merge 支持或拒绝逻辑，并在文档说明。

### ComfyUI Vendor 树

`custom_nodes/*/_vendor/` 是 live `library/` / `networks/` 的发布副本。

- 先改 live source。
- 再运行或提醒运行 `python tasks.py vendor-sync` / `make vendor-sync`。
- 不要手工让 vendor 和 live source 分叉。

## 前端实现约束

构建或修改 WebUI 前端时，必须避免生成臃肿的石山代码。

- 新增前端业务代码默认尽量控制在 1000 行以内；需求复杂、无法合理控制时，先说明原因、
  拆分方案和预计代码规模。
- 按职责拆分页面、组件、hooks、utils、api、constants、styles，遵循现有 `web/static/js/`
  和 `web/static/css/` 模块边界。
- 禁止把大量逻辑堆进单个 `App`、页面文件或超大组件；单个组件建议不超过 200-300 行，
  单个函数建议不超过 80 行。
- 重复 UI 和重复逻辑必须抽象复用，但不要为压缩行数牺牲可读性、功能完整性和可测试性。
- 样式应简洁克制，避免大段重复 CSS、硬编码结构和无法维护的视觉特殊分支。

## WebUI 维护

入口：

- 后端路由：`web/routes/config.py`、`training.py`、`settings.py`、`preview.py`、`analysis.py`。
- 业务服务：`web/services/config_service.py`、`settings_service.py`、`training_service.py`、
  `preview_service.py`、`weight_analysis_service.py`。
- 拆分业务：`web/services/config/` 和 `web/services/training/`。
- 前端 bootstrap：`web/static/app.js`。
- 前端模块：`web/static/js/`。
- DOM 锚点：`web/static/index.html`。
- 样式入口：`web/static/style.css`，具体样式在 `web/static/css/*.css`。

规则：

- `web/static/app.js` 只做 ES module bootstrap；业务放入 feature 模块。
- 当前主容器是 `web/static/js/features/anima-app/`；不要恢复 `js/features/legacy-app.js`。
- `globalThis` 只允许作为旧代码迁移桥或第三方库兼容桥；新 WebUI 业务默认使用 `export/import`
  和显式 `ctx` / store，不要新增隐式全局状态总线。
- `web/static/js/features/anima-app/chunks/` 是历史机械拆分过渡层；新功能优先放入独立 feature
  目录，修改 chunk 时优先把相关状态和函数迁出。
- 事件绑定、拖拽、筛选、弹窗、状态渲染等重复前端逻辑应抽到 shared helper 或 feature-local
  helper，不要复制一套近似 DOM 操作。
- 更新前端 import 时，同步 cache token，避免浏览器读旧模块。
- DOM id 是跨模块契约；改 `index.html` 前先搜索 selector 和相关测试。
- CSS 新文件必须从 `style.css` 可达，并遵守 import 顺序。
- `configs/web-ui-settings.toml [global]` 由 `settings_service.py` 管理，`output_root` 默认
  `output/runs`。
- runtime、history、preview 图片、队列项删除必须受 `resolve_output_root()` 边界约束。
- `memory_probe_jsonl = "auto"` 和 `block_swap_profile_jsonl = "auto"` 应解析到当前任务目录，
  不要写回用户配置固定路径。
- sample prompts 默认 `configs/sample_prompts.txt`，按配置分叉到
  `configs/sample-prompts/<methods_subdir>/<config-stem>.txt`。保留注释、空行和用户格式。
- 历史任务模式只保留 `collection` / `collections`，不要恢复旧 `config` / `flat` 模式。

常用 WebUI 验证入口见
[前端健康度评分卡](docs/features/frontend-health-scorecard.md)和
[`tests/test_training_frontend_*.py`](tests/)。后端按改动领域从 [`tests/`](tests/) 中选择
配置、预览、队列、历史或权重分析的定向测试，不要在本协议复制一份容易漂移的测试清单。

## Adapter 和 Network 维护

修改 `networks/` 前先读 `networks/CLAUDE.md`。常见定位：

- LoRA family：`networks/lora_anima/config.py`、`factory.py`、`network.py`、`loading.py`。
- 单个 LoRA 变体：`networks/lora_modules/*.py`。
- LoHa/LoKr/VeRA/GLoRA 插件：`networks/plugins/<name>/module.py` 和 `save.py`。
- IP-Adapter/EasyControl/Soft Tokens/BYG 等：`networks/methods/`。
- 保存：`networks/lora_save.py`。
- fused/split projection：`networks/attn_fuse.py`。
- attention backend：`networks/attention_dispatch.py`。

新增或修改方法时检查：

- config：`configs/methods/`、必要时 `configs/gui-methods/`。
- registry：`networks/__init__.py` 或插件注册点。
- 保存/加载 metadata 和兼容拒绝逻辑。
- 推理加载：`library/inference/adapters.py` 或方法专属路径。
- ComfyUI loader 是否需要同步。
- tests、bench、docs 和 `tasks.py` 入口。

## 训练、推理和预处理

训练：

- 入口是 `train.py`，可复用逻辑通常在 `library/training/`。
- 新 CLI 参数要检查 `library/training/cli_args.py`、config schema、WebUI、README/docs。
- WebUI/daemon 启动训练还要查 `library/runtime/launch.py`、`scripts/tasks/_common.py`、
  `web/services/training_service.py` 和 `web/services/training/*`。
- 涉及 GPU、accelerate、compile 时，测试不要依赖真实大模型；优先小 fixture 或 monkeypatch。

推理：

- adapter 加载先查 `library/inference/adapters.py` 和具体 network loading。
- sampler/denoising 改动先查 `library/inference/generation.py`、`sampling.py`、
  `sampler_context.py`。
- DCW、Spectrum、SPD、SMC-CFG、CNS、mod-guidance 应组合或明确互斥；改一个不要静默破坏另几个。
- DirectEdit 涉及 Anima Tagger 和 inversion，先读 `docs/experimental/directedit_editing_v3.md`。

预处理：

- task wrapper 在 [`scripts/tasks/preprocess.py`](scripts/tasks/preprocess.py)。
- CLI 与编排实现分别位于 [`preprocess/`](preprocess/)、[`scripts/preprocess/`](scripts/preprocess/)
  和 [`library/preprocess/`](library/preprocess/)；修改前用 `rg` 定位当前调用链，不在总协议维护脚本清单。
- 缓存路径和 sidecar 命名不要随意改；改后需要迁移说明或兼容读取。

## Daemon 和队列

- 命令入口：`tasks.py` 的 `daemon`、`daemon-attach`、`daemon-kill`、`daemon-terminate`。
- wrapper：`scripts/tasks/daemon.py`。
- daemon 实现：`scripts/daemon/`。
- WebUI 队列：`configs/web-training-queue/`。
- WebUI 历史：`configs/web-training-history/`。
- 队列失败策略、GPU 白名单、runtime config 和进度解析在
  `web/services/training/{queue,gpu,runtime_config,progress_parser,launcher,live_monitor}.py`。
- 不要硬编码 daemon 端口；发现机制以 pidfile / daemon metadata 为准。

## Custom Nodes

- 先看对应节点 `README.md`。
- Hydra loader 还要读 `custom_nodes/comfyui-hydralora/CLAUDE.md`。
- 修改 live `library/` 或 `networks/` 后，如影响发布节点，运行或提醒
  `python tasks.py vendor-sync`。
- 不要在 `_vendor/` 里做独立修复；先改源，再同步。

## 外部工具和父目录依赖

- 父目录里常见配套仓库包括 `../comfy/`、`../sam3/` 等。跨仓排障前先
  `ls ..` 确认真实存在的工具目录，不要假定所有外部依赖都已装好。
- `custom_nodes/`、ComfyUI 工作流、SAM3 和一些推理/预处理流程会间接依赖这些父目录仓库；
  改路径、说明文档或集成逻辑时要把这层依赖写清楚。

## 文档维护

- 文档入口是 `docs/README.md`。
- 根 `README.md` 只做项目介绍、部署快照和最高频入口；完整文档必须从根 README 明确链接到
  `docs/README.md`。
- 用户安装、WebUI 流程和启动命令变更：更新根 `README.md`。
- 文档索引、方法入口、坏链整理：更新 `docs/README.md`；如果分区有独立索引，也要同步更新
  对应 `README.md`。
- 新增 `docs/**/*.md` 时，必须让它从 `docs/README.md` 或一个分区索引可达。超过 5 篇文档、
  或长期增长的分区应维护自己的 `README.md`。
- 历史计划、完成报告、一次性上游合并材料、过期提案默认归档到 `_archive/docs/`，不要继续放在
  活跃 `docs/proposal/` 里。
- 活跃或半活跃提案放 `docs/proposal/`，归档时同步更新 `docs/proposal/README.md`、
  `docs/archive-index.md` 和 `_archive/docs/<subdir>/README.md`。
- 文档顶部可用状态块标注适用范围，特别是实验、历史、占位和归档文档：
  `状态：稳定 / 实验 / 历史 / 已归档 / 占位`、`适用版本：当前 main / 指定提交`、
  `入口命令：python tasks.py ...`、`相关代码：path/to/file.py`。
- 稳定能力使用说明：`docs/methods/`。
- 可运行但实验中的能力：`docs/experimental/`。
- 原理、数学、架构图解：`docs/structure/`。
- 配置、路径、环境变量和外置配置：`docs/configuration/`。
- WebUI 独立功能说明：`docs/features/`。
- 实验结论、失败路径、审计报告：`docs/findings/`。
- compile、kernel、显存和性能优化记录：`docs/optimizations/`。
- 活跃或半活跃提案：`docs/proposal/`。
- 过期或缺失上下文材料：`_archive/docs/`，并标注历史状态。
- bench 说明：对应 `bench/<method>/README.md`。
- 纯文档改动至少跑：`git diff --check -- README.md AGENTS.md docs _archive/docs`。
- 大规模文档整理还要跑本地 Markdown 链接检查，确认真实坏链为 0；外部链接只在需要时人工抽查。
- 用户可见命令优先写 `python tasks.py <command>` 或 `.venv/bin/python tasks.py <command>`；
  `make <target>` 可作为兼容说明，但不要作为唯一入口。

## 验证策略

- 后台测试默认加 `timeout 60`。
- 需要项目 Python 依赖的验证命令，默认使用 `.venv/bin/python`，避免系统 Python
  缺少 torch、pytest 插件或本仓依赖导致误判。
- 优先从 [`tests/`](tests/) 中按改动模块和文件名选择定向测试；跨模块改动再扩大到
  `timeout 60 .venv/bin/python tasks.py test-unit`。
- 大模型、真实训练、下载类验证不要默认执行；涉及 GPU 的代码优先使用小 fixture 或 monkeypatch。
- lint/format 会改文件时，只在范围明确时运行。

## 贡献规范

贡献等级、bench 证据、方法文档和 PR checklist 统一以
[`CONTRIBUTING.md`](CONTRIBUTING.md) 为准，本协议不维护第二份摘要。

## 完成前检查

- `git diff --check` 对你改过的路径干净。
- 只改了任务相关文件，没有误碰用户数据目录。
- 新行为有测试或明确说明无法低成本测试。
- 用户可见命令、配置字段、WebUI 表单、文档入口保持同步。
- 如果改了 custom nodes 相关 live source，说明是否需要 `vendor-sync`。
- 最终回复用简短中文说明改了什么、验证了什么、还有什么未做。
