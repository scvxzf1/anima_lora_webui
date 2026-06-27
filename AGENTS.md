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

## 环境和命令入口

- 项目运行环境是 Python 3.13，依赖管理优先使用 `uv`。
- `tasks.py` 是命令入口真相；`Makefile` 只是薄转发。查命令实现时读
  `tasks.py`、`scripts/tasks/` 和 `scripts/experimental_tasks/`。
- 维护和验证命令优先使用 `.venv/bin/python`；只有确认无需项目虚拟环境，或 `.venv/`
  不存在时，才回退到系统 `python`。
- 跨平台用户文档可写成 `python tasks.py <command>`，本仓维护执行优先写成
  `.venv/bin/python tasks.py <command>`。
  用户文档里常见 `make <target>`，但维护时不要只看 `Makefile`。
- `python tasks.py <command> KEY=value` 支持 Make 风格尾随环境变量，例如：
  `python tasks.py print-config METHOD=lora PRESET=default`。
- 实验命令通常以 `exp-*` 开头，可能变动或删除。改实验命令时同步检查
  `scripts/experimental_tasks/`、配置、文档和测试。
- 常用启动：
  - WebUI：`.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102`
  - GUI：`.venv/bin/python tasks.py gui`
  - 单元测试：`timeout 60 .venv/bin/python -m pytest tests/<test_file>.py`
  - 全量单测入口：`timeout 60 .venv/bin/python tasks.py test-unit`
  - 合并配置查看：`.venv/bin/python tasks.py print-config METHOD=<name> PRESET=<name>`
- 从旧 `CLAUDE.md` 继承且仍然有效的最小初始化：
  - `uv sync`
  - `hf auth login`
  - `python tasks.py download-models`
  - `python tasks.py preprocess`
  - 默认训练图片放在 `image_dataset/`，同名 `.txt` caption sidecar 也放这里。
- 命令面速记：
  - 稳定训练/服务：`lora`、`lora-gui`、`web`、`gui`、`daemon*`、`preprocess*`、
    `caption-index`、`preprocess-tagger`、`tagger`、`test-tagger`
  - 稳定推理/工具：`test`、`test-mod`、`test-hydra`、`test-merge`、`test-dcw`、
    `test-dcw-v4`、`merge`、`distill-mod`、`export-logs`、`vendor-sync`、
    `print-config`、`update`
  - 实验训练：`exp-turbo`、`exp-turbo-prep`、`exp-spd`、`exp-soft-tokens`、
    `exp-chimera`、`exp-ip-adapter`、`exp-easycontrol`、`exp-byg`
  - 实验推理/探针：`exp-test-soft`、`exp-test-turbo`、`exp-test-spd`、
    `exp-test-ip`、`exp-test-easycontrol`、`exp-test-byg`、
    `exp-test-directedit`、`exp-test-directedit-dry`、`exp-invert-directedit`
  - `python tasks.py --help` 是命令面的实时快照；老文档里出现但不在 `tasks.py`
    当前表里的命令，按历史/兼容入口处理，不要直接假定仍可用。

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

- 先分清远程语义再操作。当前仓库常见远程里，`private` 通常是个人主仓，
  `webui` 通常是发布目标，`origin` 可能是上游参考仓；推送前先 `git remote -v`
  和 `git branch -vv` 确认，不要想当然混推。
- 推送前至少检查：`git status --short --branch`、`git log <target-remote>/main..HEAD`，
  再跑和改动直接相关的测试。未跟踪文件默认不随推送发布，除非用户明确要求或本次任务
  已确认需要纳入版本控制。
- 用户说“推送更新到线上”时，先确认是推到哪个远程分支；如果上下文里已明确发布目标，
  再执行推送。推送后要汇报目标远程/分支、最新提交 hash，以及本地是否还有未提交或
  未跟踪改动残留。
- 用户说“回滚”时，先分清是哪一种：
  - 本地工作区回退：丢弃未提交改动。只有用户明确要求时才做，执行前说明会丢失哪些内容。
  - 本地提交回退：撤销本地一个或多个提交。共享分支默认优先 `git revert`，不要默认改写历史。
  - 线上回退：撤回远程分支上的已发布提交。必须先确认目标远程、目标分支和目标提交。
- 需要以线上仓库为准同步本地时，先 `fetch` 目标远程并比较 `HEAD` 和远端分支，再决定
  是否 reset。不要在没核对差异前直接做破坏性操作。
- `git reset --hard`、`git checkout -- <path>`、`git push --force`、`git push --force-with-lease`
  都视为高风险操作。除非用户已经明确要求，或已经明确给出目标提交/分支并接受影响，
  否则不要执行。
- 如果确实要改写线上历史，优先使用 `--force-with-lease` 而不是裸 `--force`，并在执行前
  说明影响范围：会覆盖哪个远程分支、抹掉哪些提交、是否影响其他协作者。
- 可以使用环境变量中的凭据或本机已配置的 SSH key 推送，但不要把 PAT、cookie、私钥、
  或带密钥的远程 URL 写进仓库文件、文档、日志样例或长期说明。

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
- `gui/`：PySide6 GUI，和 WebUI 是不同前端。
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

从 2024-06-24 起，`configs/` 目录支持通过环境变量外置到独立位置：

- `ANIMA_CONFIGS_ROOT` - 配置根目录（包含 base.toml, methods/, datasets/, web-training-history/, web-training-queue/ 等所有配置内容）

设置方法：
1. 在项目根目录创建 `.env` 文件，参考 `.env.example`
2. 或在 WebUI 全局设置面板中配置 `configs_root` 路径

优先级链（从高到低）：
1. `.anima-webui-settings.toml [paths].configs_root` - WebUI 自动管理的本机配置文件
2. `ANIMA_CONFIGS_ROOT` 环境变量
3. 默认值 `configs/`（项目根目录下）

`.anima-webui-settings.toml` 说明：
- 存放于项目根目录，WebUI "全局设置"面板保存时自动创建/更新
- 已添加到 `.gitignore`，不会提交到版本控制
- `.anima-webui-settings.toml.example` 是模板文件，供新部署参考
- 老版本部署升级后若缺失此文件，仍可通过环境变量或默认值正常工作

路径解析规则：
- 相对路径相对于项目根目录解析
- 绝对路径直接使用
- 支持 `$HOME`、`~` 等环境变量扩展
- 自动拒绝包含 `..` 的路径

详细说明：`docs/configuration/external-configs.md`

实现位置：
- `library/env.py::get_configs_root()` - 配置根目录获取（依次检查 WebUI 设置文件、环境变量、默认值）
- `library/env.py::get_training_history_root()` - 训练历史目录获取（回退到 configs_root/web-training-history）
- `library/env.py::get_training_queue_root()` - 训练队列目录获取（回退到 configs_root/web-training-queue）
- `web/services/settings_service.py::CONFIGS_DIR` - WebUI 配置目录
- `web/services/settings_service.py::GLOBAL_CONFIG_PATH_KEYS` - WebUI 可配置路径键
- `web/services/settings_service.py::_save_configs_root_override()` - 保存到 `.anima-webui-settings.toml`
- `web/services/training_service.py::HISTORY_DIR` 和 `QUEUE_DIR` - 训练服务路径

未设置任何配置时保持原有行为（`configs/` 在项目根目录），完全向后兼容。

### 训练配置合并链

```text
configs/base.toml
  -> configs/presets.toml[<preset>]
  -> configs/<methods_subdir>/<method_or_variant>.toml
  -> CLI args
```

- 默认 `methods_subdir="methods"`；WebUI/GUI 友好变体使用 `configs/gui-methods/`。
- `configs/base.toml` 包含共享基础路径、optimizer、compile flag 和默认数据集蓝图。
- `configs/presets.toml` 放硬件/采样 profile，不要把硬件 profile 复制进方法配置。
- `configs/methods/` 放算法 family 配置。
- `configs/gui-methods/` 放自包含用户变体，不使用注释切换块；实时列表以目录为准。
- `configs/datasets/` 放可复用数据集蓝图。
- `configs/imported/` 是 WebUI 导入或用户配置，默认视为用户数据。
- `configs/sample-prompts/` 放按配置分叉保存的 sample prompts。
- `configs/web-ui-settings.toml` 保存 WebUI 全局设置和模型路径；不要把本机绝对路径写进默认值或文档。
- 当前 `configs/presets.toml` 预设包括：`default`、`low_vram`、
  `low_vram_blockswap`、`balanced_16g`、`graft`、`half`、`quarter`、
  `tenth`、`debug`。
- 当前 `configs/gui-methods/` 变体包括：`chimera_hydra`、`easycontrol`、
  `glora`、`hydralora`、`hydralora-8gb`、`ip_adapter`、`loha`、`lokr`、
  `lora`、`lora-8gb`、`lora_signal_probe`、`ortholora`、`reft`、
  `soft_tokens`、`tlora`、`tlora-8gb`、`tlora_ortho_reft`、`vera`；
  维护时仍以目录实时列表为准。

默认数据和缓存：

- 源图片通常在 `image_dataset/`，同名 `.txt` 是 caption master。
- 常见产物：`post_image_dataset/resized/`、`post_image_dataset/lora/`、
  `post_image_dataset/masks/`、`output/ckpt/`、`output/tests/`、`output/runs/`。
- subset 可以设置 `cache_dir`，让 VAE/text/PE sidecar 写入专用缓存目录。
- 重要 sidecar：
  - `{stem}_{WxH}_anima.npz`：VAE latent cache。
  - `{stem}_anima_te.safetensors`：text encoder cache。
  - `{stem}_anima_pe.safetensors`：PE-Core vision feature cache。

## 方法和能力速记

- LoRA family：核心仍是三轴路由表面，默认读 `configs/methods/lora.toml`，
  再结合 `networks/lora_anima/` 和 `networks/lora_modules/`。
- Spectrum：训练外推理加速，代码在 `networks/spectrum.py`，使用和限制先读
  `docs/methods/spectrum.md`。
- DCW：采样器边界的 bias correction，校准链路在 `scripts/dcw/` 和
  `scripts/tasks/dcw.py`；用户说明在 `docs/methods/dcw.md`。
- DirectEdit + Anima Tagger：先读
  `docs/experimental/directedit_editing_v3.md` 和
  `docs/experimental/anima_tagger.md`；功能自检优先跑
  `python tasks.py exp-test-directedit-dry`。
- Modulation guidance：入口是 `python tasks.py distill-mod`，说明在
  `docs/methods/mod-guidance.md`。
- IP-Adapter：训练入口 `exp-ip-adapter`，推理入口 `exp-test-ip`，文档在
  `docs/experimental/ip-adapter.md`。
- EasyControl：训练入口 `exp-easycontrol`，推理入口 `exp-test-easycontrol`，
  文档在 `docs/experimental/easycontrol.md`。
- Soft Tokens：训练入口 `exp-soft-tokens`，推理入口 `exp-test-soft`，文档在
  `docs/experimental/soft_tokens.md`。
- ChimeraHydra：训练入口 `exp-chimera`，结构/实验文档看
  `docs/experimental/chimera-hydra.md` 和 `docs/structure/chimera-hydra.md`。
- Postfix 和 postfix-tail inversion：当前更多是实验/兼容入口。先读
  `docs/experimental/postfix.md`、`docs/guidelines/training.md#postfix`、
  `docs/proposal/postfix_residual_per_image_inversion.md`，再碰
  `scripts/inversion/` 或 `library/inference/editing/postfix_inversion.py`。

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

不要把 DiT 提前加载到预处理阶段。WebUI、task runner、daemon 和 GUI 启动训练时也要保持
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

构建或修改 WebUI / GUI 前端时，必须避免生成臃肿的石山代码。

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

常用 WebUI 验证：

- 前端模块图、DOM、事件钩子、CSS import：
  `timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py`
- sample prompts/config：
  `timeout 60 .venv/bin/python -m pytest tests/test_web_config_service.py`
- preview/global settings/output root：
  `timeout 60 .venv/bin/python -m pytest tests/test_preview_service.py`
- 队列/runtime 安全：
  `timeout 60 .venv/bin/python -m pytest tests/test_training_queue.py`
- 权重分析：
  `timeout 60 .venv/bin/python -m pytest tests/test_weight_analysis_service.py`

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
- 新 CLI 参数要检查 `library/training/cli_args.py`、config schema、WebUI/GUI、README/docs。
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

- task wrapper 在 `scripts/tasks/preprocess.py`。
- CLI 脚本在 `preprocess/` 或 `scripts/preprocess/`。
- 编排逻辑优先查 `library/preprocess/`。
- 缓存路径和 sidecar 命名不要随意改；改后需要迁移说明或兼容读取。
- 旧 `CLAUDE.md` 里常提到、现在仍是常用入口的预处理脚本有：
  `scripts/preprocess/resize_images.py`、`cache_latents.py`、
  `cache_text_embeddings.py`、`cache_pe_encoder.py`、`generate_masks.py`、
  `generate_masks_mit.py`、`merge_masks.py`。
- 其他常用脚本入口：
  `scripts/merge_to_dit.py`、`scripts/comfy_batch.py`、
  `scripts/export_logs_json.py`、`scripts/edit.py`、
  `scripts/anima_tagger/cli.py`、`scripts/dcw/`、`scripts/inversion/`。

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
- 用户安装、WebUI 流程和启动命令变更：更新根 `README.md`。
- 文档索引、方法入口、坏链整理：更新 `docs/README.md`。
- 稳定能力使用说明：`docs/methods/`。
- 可运行但实验中的能力：`docs/experimental/`。
- 原理、数学、架构图解：`docs/structure/`。
- 实验结论、失败路径、审计报告：`docs/findings/`。
- compile、kernel、显存和性能优化记录：`docs/optimizations/`。
- 活跃或半活跃提案：`docs/proposal/`。
- 过期或缺失上下文材料：`_archive/docs/`，并标注历史状态。
- bench 说明：对应 `bench/<method>/README.md`。
- 纯文档改动至少跑：`git diff --check -- docs _archive/docs`。如果只改根说明文件，
  跑 `git diff --check -- AGENTS.md` 即可。

## 验证策略

- 后台测试默认加 `timeout 60`。
- 需要项目 Python 依赖的验证命令，默认使用 `.venv/bin/python`，避免系统 Python
  缺少 torch、pytest 插件或本仓依赖导致误判。
- 优先跑和改动直接相关的 pytest 文件或测试名。
- 大模型、真实训练、下载类验证不要默认执行。
- lint/format 会改文件时，只在范围明确时运行。

常见选择：

- Web config/sample prompts：`tests/test_web_config_service.py`、`tests/test_config.py`。
- Web global settings/preview paths：`tests/test_preview_service.py`。
- Web training queue/runtime：`tests/test_training_queue.py`、`tests/test_training_resume.py`。
- Web frontend modules/history/preview hooks：`tests/test_training_frontend_state.py`。
- Web weight analysis：`tests/test_weight_analysis_service.py`。
- daemon/CLI launch：`tests/test_daemon.py`、`tests/test_runtime_harness_cli.py`、
  `tests/test_launch_config.py`。
- preprocess：`tests/test_preprocess_dataset.py`、`tests/test_preprocess_paths.py`。
- network registry/config/metadata：`tests/test_network_registry.py`、`tests/test_network_cfg.py`,
  `tests/test_method_network_lifecycle.py`、`tests/test_factory_metadata_flow.py`。
- LoRA/LoHa/LoKr/VeRA/GLoRA：`tests/test_lora_custom_autograd.py`、`tests/test_loha.py`、
  `tests/test_lokr.py`、`tests/test_vera.py`、`tests/test_glora.py`。
- Hydra/FeRA/Chimera routing：`tests/test_global_router.py`、`tests/test_router_compute.py`、
  `tests/test_hydra_sigma_band.py`、`tests/test_fera_fecl_handler.py`、
  `tests/test_chimera_router_stats.py`。
- inference/editing：`tests/test_generation_request.py`、`tests/test_edit_dispatcher.py`、
  `tests/test_directedit_v_injection.py`、`tests/test_experimental_inference_tasks.py`。
- training basics：`tests/test_training_bootstrap.py`、`tests/test_training_optimizers.py`、
  `tests/test_training_resume.py`、`tests/test_training_gpu_selection.py`。
- text strategy / bucket invariants：`tests/test_ensure_text_strategies.py`、
  `tests/test_constant_token_buckets.py`、`tests/test_native_flatten.py`。
- tagger/captions：`tests/test_anima_tagger_dual_encoder.py`、`tests/test_caption_index.py`、
  `tests/test_caption_shuffle.py`、`tests/test_tag_groups.py`、`tests/test_tag_taxonomy.py`。

## 贡献等级

参考 `CONTRIBUTING.md`：

- Tier 1：bug、UI、CLI 小修。保持范围小，跑相关测试。
- Tier 1.5：效率、显存、数值或现有算法修改。需要 bench 脚本、invariant test、文档和兼容性说明。
- Tier 2：新 adapter/method。需要论文或明确依据、`bench/<method>/`、docs、tests、
  `tasks.py`/Make 入口和 merge story。
- Tier 3：新 base model 当前不接受。只做文档或对 Anima 本身有价值的解耦工作。

## 完成前检查

- `git diff --check` 对你改过的路径干净。
- 只改了任务相关文件，没有误碰用户数据目录。
- 新行为有测试或明确说明无法低成本测试。
- 用户可见命令、配置字段、WebUI 表单、文档入口保持同步。
- 如果改了 custom nodes 相关 live source，说明是否需要 `vendor-sync`。
- 最终回复用简短中文说明改了什么、验证了什么、还有什么未做。
