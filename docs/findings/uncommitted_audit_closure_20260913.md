# 未提交更新审计收尾（2026-09-13）

状态：阶段性收尾；定向验证通过，完整 Core 门禁未通过，尚不作为发布批准。

分支：`codex/audit-uncommitted-20260913`；基线：`origin/dev` 的 `2e96bd0c`。
用户授权修正后推进本地提交，本轮不推送、不发布静态构建、不启动训练。

## 已核实和修正

- raw rename HTTP 入口拒绝非对象 JSON、非字符串和空白路径，返回 400；新增 11 项真实 aiohttp 契约测试。
- Next 配置库键盘排序按列表相对位置决定 before/after，指针仍按目标中线判断；新增向下排序成功 E2E，确认单次 POST 与服务端顺序。
- Classic/Dragon 同一 ESM 目标统一 query token，避免重复模块单例；不要求所有不同模块共享一个版本号。
- 同步 7 项配置帮助摘要与详细说明，恢复 base_compute 的冻结底模与 NF4/block swap 边界；更新过期字段计数、模块路径和 token 断言。
- 修正历史日志 mock、配置库入口/坐标/请求体断言和 caption 对话框双重处理竞态；修正 preflight 时长 fixture 与参数语义测试 facade 同步。
- 文档保留模型族“允许尝试”与“已验证”区别，补充实验生命周期标签；旧前端审计保留为修正前快照。

撤回的推断：常规 dataset blueprint 已通过 loader fallback 传播 `args.model_family`，无需重复修补；registry 开放 variant 是当前设计，不恢复 blanket plain-LoRA gate；CSS 使用 no-cache，旧 token 本身不能证明缓存失效，实际修复针对 ESM URL 身份重复。

## 提交与验证

代码按显式路径白名单暂存，并导出 index 快照执行 Python 定向测试，避免依赖未暂存源码。

| 提交 | 范围 | 定向证据 |
| --- | --- | --- |
| `1b01b53` | 自适应预处理批次、bucket 尺寸 | 75 passed |
| `2737308` | family cache、动态 block swap、训练接线 | 281 passed |
| `0e27ff5` | NF4 DoRA、OrthoLoRA、ReFT 契约 | 58 passed |
| `86e1582` | swap 校准与 pipeline 实验探针 | 65 passed |
| `4224ac4` | Web 配置、历史、队列及分页 API | 304 passed |
| `60e4293` | Classic/Dragon 配置工作台及模块契约 | index 首轮 201 passed / 5 skipped / 3 failed；补齐依赖文档后对应 5 项通过；Classic 修正定向组 22 passed |
| `1d1e6b5` | Next 工作台、历史、caption、mask 与命令入口 | index Python 4 passed；Vitest 46 文件 / 210 passed；TypeScript、Vite build 通过 |

Classic 完整运行曾为 364 passed / 4 failed，4 项已修正并通过定向复验，未再完整重跑，不宣称整套全绿。

Next 完整 E2E 为 127 passed / 1 failed / 2 skipped；唯一失败是 caption 对话框测试竞态。修正后 caption、配置库排序/命名、历史返回四个文件共 21 项通过，包含完整运行收集后新增的键盘排序成功用例。完整套件未再重跑。两项 skip 为已有 dataset-drag 测试。隔离 Vite 使用 mock API 与不可达后端目标；未写真实训练数据。配置库桌面/手机截图已检查，无空白或明显重叠。

构建输出位于 `/tmp/krea2-audit-next-build-final-20260913`，没有通过发布命令覆盖生产静态目录。

## 未关闭风险和保留项

- Core 全套在 300 秒超时；随后 fail-fast 运行得到 873 passed / 96 deselected / 1 failed。失败为 `test_dynamic_block_swap_runtime.py::test_nf4_unindexed_cuda_reuses_storage_and_quant_state`，bitsandbytes 报 CPU backend 缺少 `cquantize_blockwise_fp32_nf4`。隔离测试通过，但未定位整套进程的初始化差异；CUDA mock 污染仅是假设，不能作为根因结论。没有用 skip 或生产代码变更掩盖失败。
- 新增真实 CUDA 测试有可用性 skip，但仍需核查 hardware 分层标记；未完成整套硬件与长训练验收。
- ComfyUI vendor 发布副本未同步；发布这些节点前仍需运行 vendor-sync 并验收。
- 大规模既有工作区更新以现有模块拆分和训练 facade 接线整理提交，本轮没有进一步重构热点文件。后续继续把 chunk 业务迁至 feature，并缩减训练编排文件；上述定向验证不等同全模型数值或长期性能保证。
- `configs/web-ui-settings.toml` 含本机模型路径，未提交；当前本机覆盖文件仅支持 paths，未擅自搬迁模型设置。
- 10 个数据集配置的工作区删除未暂存；导入 sample prompts、`.cursor` 调试日志、根目录 `torch` 文件及手工 mask server 未提交，均原样保留。
- 没有提交模型、输出、队列、历史、缓存或 node_modules，也没有执行推送。
