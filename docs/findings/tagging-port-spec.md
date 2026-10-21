# 打标能力净室移植行为契约（tagging port spec）

状态：Phase 1 / Phase 2 已实现、已审计、已复核（2026-10-21；含 t7 收尾的 captions_json fail-closed 与 skip 预筛后缀对齐、t14 的嵌套布局守卫对齐，均已独立复核并覆盖当前 revision）
适用版本：当前 `dev`
入口命令：`python tasks.py web --host 127.0.0.1 --port 20102`
相关代码：`web/services/tagging/caption_snapshots.py`、`web/services/tagging/caption_edit.py`、`web/services/tagging/storage.py`、`web/services/tagging/jobs.py`、`web/routes/tagging.py`、`web/frontend-next/src/features/captioning/`

审计链（2026-10-21）：后端只读审计 12 条（BAC-1…BAC-12）、前端只读审计 12 条（FAC-1…FAC-12），
修复后由独立复核给出 `verdict=pass`（round 2 复核：t11 / t13 / t15）。逐条最终状态见 §5，遗留项见 §6。

## 0. 来源与净室声明

本契约用于把外部项目 `AnimaLoraStudio`（GPL-3.0）打标子系统的**行为**引入本仓库
（MIT），采用净室重写路线：

- **只取行为**：输入输出、边界条件、失败语义、数据布局这类思想与功能不受版权保护。
- **不取表达**：不复制其源码文本、注释、docstring、prompt 文本、测试用例代码与 fixture 数据。
- **实现独立**：本仓库按自己的架构（aiohttp + `web/services/tagging/` + dataset 预设）重新设计，
  文件切分、命名、错误类型与存储格式均不相同。
- 行为参考来自打标功能清点报告（`2026-10-21-tagging-feature-inventory-v0.29.0.md`）与对目标仓
  源码的阅读理解；`caption_snapshots.py` 对应其 `caption_snapshot.py` 的**语义**，不含其实现。

依赖边界：本移植**不新增任何第三方依赖**（`pyproject.toml` 与锁文件零 diff）。

## 1. 架构差异（为什么不能整块搬）

| | 目标仓 | 本仓 |
|---|---|---|
| 打标模型 | 项目版本级一次性全量打标落盘 | **dataset 候选式**：job 产出 `proposed_caption` 候选，人工审核后 commit 落盘 |
| 打标范围 | `<version_dir>/train/` 全部子目录 | dataset 预设行解析出的单个图片目录（`source_dir` 或 `image_dir`） |
| 快照范围 | train/ 全量 caption 打包 zip | 当前 dataset 行的 caption sidecar 集合 |
| 快照位置 | `<version_dir>/caption_snapshots/{ts}.zip` | `configs/captioning/snapshots/<id>.json` |
| 还原语义 | 清空 train/ 全部 caption 后解包 | 仅还原快照范围内、与图片同名的 caption sidecar |

因此本契约只继承「**写入前先建还原点**」这一安全属性，不继承其目录结构与调用链。

## 2. Phase 1 契约：caption 还原点

### 2.1 存储布局

```text
<configs>/captioning/snapshots/<snapshot_id>.json
```

- `<configs>/captioning` 由 `web/services/tagging/settings.py::SETTINGS_FILE` 的父目录决定，
  与 tagging 设置 / 预设 / 词典沿用同一运行时目录（`configs/captioning/` 已被 `.gitignore` 忽略）。
- `snapshot_id` 形如 `20261021-101530-a1b2c3`，允许字符为 `[0-9A-Za-z_-]`，长度 8–64；
  任何含路径分隔符、`..` 或不匹配的 id 一律拒绝。
- 单文件 JSON，通过 `web/services/atomic_io.py::atomic_write_text` 原子落盘；读取时若 JSON 损坏，
  列表接口返回带 `broken=true` 的条目而不是整体失败；`restore`/`delete` 遇到损坏文档或
  不支持的 `schema` 版本则抛 `CaptionSnapshotError`（400），不会尝试修复式还原。

快照文件结构（schema v1）：

```json
{
  "schema": 1,
  "id": "20261021-101530-a1b2c3",
  "created_at": "2026-10-21T10:15:30+08:00",
  "created_at_ms": 1761006530000,
  "reason": "commit",
  "label": "",
  "dataset_file": "configs/datasets/example.toml",
  "dataset_index": 0,
  "source": "source",
  "scope_dir": "/abs/path/to/images",
  "totals": {"files": 3, "bytes": 128},
  "entries": [
    {"name": "a.txt", "size": 12, "sha256": "…", "encoding": "utf-8", "data": "1girl, solo"}
  ]
}
```

- `entries[].name` 必须是**裸文件名**（不含分隔符）；`encoding` 为 `utf-8`（`data` 为文本）
  或 `base64`（`data` 为 base64，用于非 UTF-8 内容），保证逐字节还原。
- `reason` 取 `commit`（写回前自动创建）或 `manual`（用户手动备份）。
- 对外摘要（列表 / 创建 / 还原返回）不含 `entries`，把 `totals.files`/`totals.bytes`
  展开为 `file_count`/`bytes`，并保留 `scope_dir` 与 `source`。

### 2.2 范围规则

`scope_dir` = 该 dataset 行在当前 `source` 下的图片根目录（`source` → `source_dir`，
`training` → `image_dir`）。

纳入快照的文件必须**全部满足**：

1. 位于 `scope_dir` 顶层（不递归子目录）；
2. 后缀属于 `{.txt, .caption, .json}`（比较时统一小写）；
3. 文件名是 `captions.json`（**大小写不敏感**，`CAPTIONS.JSON` 同样算聚合 caption），
   或其主干名与同目录下某个真实图片文件的主干名相同。

即：只有「与图片配对的 caption sidecar」会被快照与还原，目录里的其它文本 / JSON 不被触碰。
快照与还原使用同一套范围规则，因此「快照里没有的文件」在还原时就是「快照后新增」而会被清除。

上限：单快照最多 `20000` 个文件、`64 MiB` 内容；超过时抛
`CaptionSnapshotTooLarge`，不写半成品文件。

保留裁剪：`max_retained`（默认 20）只裁剪**同一 `dataset_file` + `dataset_index`** 下的旧快照；
`dataset_file` 为空时直接跳过裁剪，避免跨数据集误删。

### 2.2.1 已知限制：快照只覆盖 scope_dir 顶层

- 还原点范围规则（2.2 规则 1）只扫描数据集图片目录的**顶层** caption sidecar，不递归子目录。
- 数据集行若按递归方式扫描子目录（`recursive` 默认开启），sidecar 来源的写回仍会落到 `<sub>/<stem>.txt`（写入目标跟随图片所在目录；`captions_json` 等聚合来源行按 §3.1 fail-closed 直接拒绝，不写任何 sidecar），但这些子目录 sidecar **不在还原点覆盖内**，`restore_snapshot` 也不会删除或回写它们。
- 需要覆盖嵌套目录时应单独评审：文件数/64 MiB 上限预算、还原范围规则与「快照后新增文件即删除」的删除语义都会随递归变化，不能作为顺手改动。
- 当前行为由 `tests/test_tagging_caption_snapshots.py::test_snapshot_scope_stays_top_level_for_recursive_dataset_layouts` 固定。

### 2.3 操作语义

| 操作 | 语义 |
|---|---|
| `create_snapshot(...)` | 扫描范围 → 收集内容 → 原子写 JSON → 在**同一 dataset 范围**内按 `max_retained`（默认 20）裁剪最旧快照 → 返回摘要（不含 `entries`） |
| `list_snapshots()` | 返回全部摘要，按 `created_at_ms` 倒序；损坏文件不进异常，标记 `broken` |
| `restore_snapshot(id)` | 见下 |
| `delete_snapshot(id)` | 删除单个快照；不存在则 `CaptionSnapshotNotFound` |

**还原算法（先全量校验、再动盘、失败整单拒绝）**：

1. 校验 id 形状（正则 + `<snapshot_root>/<id>.json`）与文档可解析性、`schema == 1`。
   不存在 → `CaptionSnapshotNotFound`（404）；损坏或版本不符 → `CaptionSnapshotError`（400）。
2. 校验 `scope_dir`：必须非空、绝对路径、当前仍存在且是目录，否则 `CaptionSnapshotError`，
   且在动盘前中止。
3. 生成**还原计划**：`entries` 必须是列表；每条必须是对象、裸文件名（无 `/`、`\`、NUL、`.`/`..`）、
   后缀 ∈ `{.txt, .caption, .json}`、`encoding` ∈ `{utf-8, base64}`、`data` 必须可解码、
   目标 resolve 后仍位于 `scope_dir` 内、文件名不得重复。任一条不合法 → 整单拒绝
   （`CaptionSnapshotError`，400，**0 删除 0 写入**）。
4. 计划全部通过后才动盘：先删除「当前范围中、不在计划里」的 caption（快照后新增），计入
   `removed`；再按计划逐条写回（`utf-8` 走 `atomic_write_text`，`base64` 走原子二进制替换），
   计入 `written`。
5. 返回 `{"ok": true, "id": …, "written": n, "removed": n, "skipped": [], "snapshot": 摘要}`。
   **`skipped` 仅为响应形状兼容保留，恒为空数组**：非法条目不再逐条跳过，而是整单拒绝。

**不变量**：还原只可能影响 `scope_dir` 顶层、且通过 2.2 范围规则的文件；不移动、不删除图片，
不触碰范围外目录；整单校验失败时磁盘零变化（没有「删了一半」的中间态）。
嵌套目录（递归数据集子目录）里的 sidecar 不在任何还原点内，见 §2.2.1。

### 2.4 写回集成

`TaggingJobManager._commit_locked()` 在写任何 caption **之前**创建 `reason="commit"` 的快照：

- 候选为空时不建快照、不写盘。
- 候选在进入快照与写回循环**之前**冻结成 `(item, 文本)` 列表，写回只使用冻结文本；
  写回过程中的人工编辑或重跑都无法改变本次写入内容。
- 成功后把摘要放进 commit 返回值的 `snapshot` 字段，并写一条 `commit_snapshot` 日志。
- 抛 `CaptionSnapshotTooLarge`（范围过大）时**不阻塞写回**，返回 `snapshot: null` 与
  `snapshot_error`，并写 warning 日志 —— 大目录不能被还原点机制卡死。
- 其它快照失败（`CaptionSnapshotError` / 磁盘 IO 错误）视为真实故障：**中止本次写回**，
  不写任何 caption，向上抛 `CaptionSnapshotError`（`ValueError` 子类），路由映射为 400。
- 例外（见 §6.2 BAC-8）：数据集行 / 目录解析失败（`FileNotFoundError`、非快照类 `ValueError`）
  仍会继续写回，但 `snapshot: null` 与 `snapshot_error` 会透出到 API 与 UI。

**并发与锁**（BAC-1 修复后的最终行为）：

- 每个 job 一把 job 级锁 `_job_locks`；`commit`、`rerun`、`update_items` 共用同一把锁
  （`_commit_locks` / `_rerun_locks` 是同一 mapping 的别名，保留旧名只为兼容既有调用）。
- commit 持锁期间：`rerun` 立即抛 `RuntimeError`（路由 409，「任务正在写回或重跑…」），
  `update_items` 同样被拒；不会再出现「写回循环读到被清空的候选」。
- commit 拿到锁后会**重新读取 job 状态**再判断，避免等锁期间 rerun 已完成造成误判。
- 同一 job 的 commit 串行；不同 job 之间没有共享锁（见 §6.2 BAC-10）。

### 2.5 HTTP 契约

两条前缀 `/api/captioning` 与 `/api/tagging` 都注册（与既有路由一致）：

| 方法 | 路径 | 请求 | 成功响应 | 失败 |
|---|---|---|---|---|
| GET | `/caption-snapshots` | 查询参数 `dataset_file`、`dataset_index`（可省，省略即不过滤） | 200 `{"ok": true, "snapshots": [摘要…]}` | 400 `dataset_index` 非整数 |
| POST | `/caption-snapshots` | `dataset_file`、`dataset_index`、`source`、`label?` | **201** `{"ok": true, "snapshot": 摘要}` | 400 参数错 / 预设或图片目录不存在 / `dataset_index` 越界；409 服务忙 |
| POST | `/caption-snapshots/{snapshot_id}/restore` | — | 200 还原结果 | 404 快照不存在 / 400 文档损坏或校验失败 |
| DELETE | `/caption-snapshots/{snapshot_id}` | — | 200 `{"ok": true, "deleted": id}` | 404 不存在 |

请求字段 `source` 取 `source`（原始图）或其它值（按 `training`，训练图）处理，缺省 `source`；
`label` 截断到 80 字符。

错误映射按实现：`KeyError` → 404，`ValueError`/`OSError` → 400（`CaptionSnapshotError` 是
`ValueError` 子类；`CaptionSnapshotNotFound` 单独映射 404），`RuntimeError` → 409，
服务未初始化 → 503。

### 2.6 前端还原点面板

打标任务审核视图（`CaptionReview`）在「写回」工具栏下方提供可折叠的**还原点**面板，
支持立即备份、还原（二次确认）与删除。

- 面板文案是**条件表述**：「写回时会先尝试创建还原点：dataset 范围过大或目录不可用时可能没有
  还原点，届时审核页会明确提示」，不再声称每次写回都必然有还原点。
- commit 结果区：`snapshot_error` 非空时渲染 `role=alert` 的「本次写回没有还原点：…」；
  有快照时显示「已创建还原点 <id>」。commit 成功后立即失效快照列表查询，新还原点无需重开页面。
- 列表查询失败时渲染 `query.error` 并提供「重试加载」；「立即备份」不因列表失败而禁用。
- 还原是磁盘操作，任务候选仍显示该 job 上次写回的内存状态；如需刷新磁盘真值请重新打开对应数据集。

## 3. Phase 2 契约：caption 写入语义

实现位于 `web/services/tagging/caption_edit.py`，写入入口是
`storage.write_caption(..., policy=..., trigger_word=...)`。

### 3.1 权威 sidecar 与有效判据

- 权威顺序：`<stem>.json` > 配置的 text 后缀（默认 `.txt`）> `.caption` > 新建 text 后缀文件。
- `has_effective_caption(image, text_suffix=…)`：JSON 看渲染文本**或** `meta.trigger`；
  text 看非空 tag 列表。空 / 纯空白 / 损坏一律视为未打标。
- `read_caption(image)` 返回可编辑 tags（JSON 按 `fixed → appearance → tags → environment` 展开）
  与渲染文本。
- 检测与写盘必须使用**同一个 `text_suffix`**：`storage.caption_text_suffix(settings)` 是唯一来源，
  `write_caption` 与 `jobs.create` 的 skip 预筛都从它取后缀（BAC-5 已对齐），不会再出现
  「判据看 `.txt`、写盘落 `.tags`」的分裂。

**聚合来源 fail-closed（BAC-4，设计决策，不要回退）**：`write_caption` 在读写任何 sidecar **之前**
调用 `_assert_sidecar_write_allowed`：

| dataset 行设置 | 行为 |
|---|---|
| `caption_source_mode=captions_json`（别名 `captions.json`/`captions-json`/`diffpipeforge`） | **拒绝写入**，抛 `AggregateCaptionWriteError`（`CaptionSidecarError` → `ValueError` 子类），提示切换为 sidecar 模式或直接编辑 `captions.json`；不新建、不修改任何 sidecar，聚合文件字节不变 |
| `auto`（默认）且**读取链会优先读取的** `captions.json`（图片所在目录 **或** dataset 根目录）已包含该图片条目 | 同样拒绝：读取链会优先读聚合条目，写 sidecar 会形成双真值 |
| 显式 `txt` / `json` | 正常写该 sidecar（聚合条目不会遮蔽读取） |
| `auto` 且图片目录与 dataset 根都没有聚合文件、或都不含该图片 | 正常写 sidecar，避免过度阻断 |

守卫的目录集合与读取链**同源**：`_caption_scope_directories(image, row)` 返回
`[image.parent, row.source_dir, row.image_dir]`（与 `read_caption_for_image` 一致），auto 模式直接用
`read_caption_source_from_dirs` 的 `detected_mode == captions_json` 判定。因此嵌套布局
（`dataset/sub/img.png` + 根 `captions.json={"sub/img.png": …}`）同样 fail-closed：零 sidecar、聚合字节不变。

理由：工作台只认 sidecar，而预览/训练读取链在聚合模式下优先读 `captions.json`；如果继续新建
`<stem>.txt`，同一张图就有两份互相看不见的真值。**这是产品决策，不是缺口**——把它当 bug 改回
「照旧写 sidecar」会立即恢复聚合与 sidecar 并存的分裂。若将来要支持直接写聚合文件，应作为新功能
单独评审（见 §6.1 BAC-4 的「不要回退」说明）。

### 3.2 写入规则

| 情形 | 行为 |
|---|---|
| 权威 sidecar 是 JSON | 保留 `meta` 与所有未知字段；移除被吸收的标签键（`tags`/`appearance`/`environment`/`quality`/`count`/`character`/`series`/`artist`/`fixed`/`from_path`/`ai_output`/`nl`）；写回顶层 `tags` |
| JSON 中的 `nl` | 被当作结构性字段先移除，再按规则条件写回：编辑后的 tag 文本**未包含**该 `nl` 时保留为 `nl` 字段（渲染为 `tags. nl`）；已包含则丢弃，避免重复 |
| 权威 sidecar 是 text | 用 `", "` 连接 tags 覆盖写入 |
| tags 为空 | 写入空内容但**保留 sidecar 文件**，不算已打标（commit 编排层的额外判据见 3.3） |
| JSON 损坏 | 抛 `CorruptCaptionError`，**拒绝写入**，不覆盖用户数据 |
| 目标路径 | 沿用 `_assert_caption_target` 的数据集目录边界校验（symlink 越界被拒） |

### 3.3 写入策略与提交语义

- `on_existing`：`overwrite`（默认）/ `skip` / `append`（tag 级保序去重合并，现有在前）。
- 任务创建时：`skip` 按 sidecar 判据**预筛**掉已有有效 caption 的图片（`captions_json` 行不在此列，
  见 §3.1）；全部被筛掉时报「所选图片都已有 caption，按『跳过已有』策略没有可处理的图片」。
- 提交范围：选中项（或 `all_items`）中文本非空、且状态允许写回的条目
  （`ready`/`failed`，或 `committed` 但磁盘文本与候选不同）。候选文本在写回前冻结。
- 候选解析后没有任何 tag（例如 `" , "`）：**不调用** `write_caption`，计入 `failed`，
  写 `commit_status="failed"` 与 `commit_error`，绝不清空已有 caption。
  （`write_caption` 自身「空 tags 写空内容、保留 sidecar」的契约不变，缺口在编排层判据。）
- 策略跳过（`skip` 命中已有 caption）：不写盘，计入 `skipped`，写 `commit_skipped` 日志。
- 聚合来源拒绝（BAC-4）：`write_caption` 抛 `AggregateCaptionWriteError`，按写失败处理——计入 `failed`、
  写 `commit_status="failed"` 与 `commit_error`（中文可操作提示），不写任何 caption sidecar。
  注意 skip 预筛只按 sidecar 判据，所以 `captions_json` 行的图片不会被预筛掉，仍会进入候选池，
  最终在提交时以 `failed` 呈现；这是 fail-closed 的直接后果（宁可失败，也不产生双真值）。
- 幂等：重复 commit 得到 `written=0`、`failed=0`，且不会重复创建快照；无候选时不建快照。

**commit 响应与计数语义（2026-10-21 收窄）**：

| 字段 | 含义 |
|---|---|
| `written` | 实际写盘成功数 |
| `conflicts` | 写回冲突数（当前实现无 `CaptionWriteConflict` 抛出点，恒为 0，见 §6.2） |
| `skipped` | **仅**「按策略跳过」的数量（`skip` 策略命中已有 caption） |
| `failed` | 写盘失败（`OSError`/`ValueError`）与「候选解析后无有效标签」的数量（新增字段） |
| `errors` | `[{item_id, file, error}]` 明细 |
| `snapshot` / `snapshot_error` | 还原点摘要 / 无还原点原因（见 2.4） |
| `job` | 写回后的任务快照 |

每个 item 新增 `commit_status` ∈ `{written, skipped, failed, conflict}` 与 `commit_error`。

这是审计发现 BAC-7 / FAC-5「写失败与策略跳过混计」被认定为缺陷后的修正，属于
**加法 + 语义收窄**：新增 `failed`/`commit_status`，`skipped` 不再包含写盘失败。
前端已同步：结果区显示「写入 N · 冲突 M · 跳过 K · 失败 F」，跳过项用弱化 note + `role=status`，
失败 / 冲突用 `form-error` + `role=alert`。兼容性注意：只读 `skipped` 的旧客户端会把写失败少算，
必须同时读取 `failed`。

- `trigger_word`：非空时作为第一个 tag 幂等 prepend（大小写不敏感去重）；含逗号 / 换行的
  trigger 会先按 tag 分隔符拆分，因此多次写入仍是恒等；JSON 同时写 `meta.trigger`。上限 200 字符。

### 3.4 与目标仓的刻意差异

- **不新增磁盘级批量算子**。目标仓的 add / remove / replace / dedupe / stats + scope 三形态会直接
  改盘；本仓是「候选 → 人工审核 → commit」的工作台，批量编辑继续留在审核草稿层
  （`captionBatch.ts`、`batch_update.py`），避免绕过人工确认。写入前的安全网由 Phase 1 还原点提供。
- **不引入覆盖率门禁**。本仓没有目标仓的 phase 状态机；「是否已有 caption」只作为策略筛选与
  界面信息使用，不阻断任何流程。

## 4. 验收与洁净度证据

以下数字为 2026-10-21 **round-2 复核（t11 / t13 / t15）之后的当前 revision** 实测结果
（复核者与本文档任务各自实跑，命令与输出随条目列出）。

- 定向后端：`tests/test_tagging_caption_write.py` + `test_tagging_caption_snapshots.py` +
  `test_tagging_caption_edit.py` + `test_tagging_service.py` → **95 passed, 2 warnings**。
- 契约清单扩展（再加 `test_tagging_routes.py` / `test_tagging_job_cleanup.py` /
  `test_tagging_batch_update.py`）→ **115 passed, 9 warnings**。
- 打标域合计：12 个 `tests/test_tagging_*.py` → **171 passed**，加上
  `tests/test_ui_scale_settings.py` 的 4 个用例，合计 **175 passed**。
- 前端：`npx vitest run src/features/captioning` → **19 files / 49 tests passed**；
  `npx tsc -b --pretty false` → **无输出（exit 0）**。
- 静态检查：`ruff check web/services/tagging/ web/routes/tagging.py` → **All checks passed!**；
  `git diff --check` 对改动路径干净。
- 依赖：`pyproject.toml`、`uv.lock`、`package.json`、`pnpm-lock` 零 diff（未新增依赖）。
- 独立复核（三条均为 `verdict=pass`，覆盖当前 revision）：
  - **t11（round 2，覆盖 t12/t14 之后的当前字节）**：t3 的 BAC-1/2/3 与 FAC-1 仍成立；t3 + t7 叠加边界
    （聚合行 skip 仍进候选池、commit 记 `failed`、拒绝前仍创建还原点）与 §3.3 一致；4 个快照端点
    双前缀 8 组状态码全对。
  - **t13（round 2，范围：t12 产物）**：快照 docstring 与行为钉桩用例真实生效；T9-1 当时作为合同外
    残留转出（后续由 t14 修复）。
  - **t15（T9-1 修复）**：嵌套布局 fail-closed 生效（零 sidecar、聚合字节不变），11 组嵌套/边界探针
    满足「拦截 ⇔ `reader.detected_mode == captions_json`」，且无过度阻断。

Phase 1 / Phase 2 定向测试覆盖：

- `tests/test_tagging_caption_snapshots.py`：范围规则、往返还原、快照后新增文件被清除、
  id 非法拒绝、上限拒绝、保留裁剪、损坏快照容错、manifest 结构损坏时删除前中止、
  `CAPTIONS.JSON` 大小写、commit 前自动快照与失败策略、递归数据集下快照范围保持顶层
  （`test_snapshot_scope_stays_top_level_for_recursive_dataset_layouts`，见 §2.2.1）。
- `tests/test_tagging_caption_edit.py`：标签切分 / 合并 / 触发词 / sidecar 优先级 /
  有效判据矩阵 / JSON 保留与吸收 / `nl` 去重 / 空标签保留 / 损坏拒绝。
- `tests/test_tagging_caption_write.py`：三种策略、触发词幂等、commit 策略透传、skip 预筛、
  写回候选冻结与并发 rerun 拒绝、失败不混入跳过计数、无标签候选不清空既有 caption、
  自定义 `caption_extension` 下预筛与写盘同判据（BAC-5）、`captions_json` 行与 auto+聚合条目
  的 fail-closed（不新建 sidecar、聚合字节不变）、显式 sidecar 模式不被误拦、
  聚合拒绝在 commit 中计入 `failed` 而非 `skipped`（BAC-4）、
  嵌套布局（`dataset/sub/img.png` + dataset 根或图片所在目录的聚合文件）fail-closed 且显式 `txt`/`json` 不被误拦（T9-1）。

净室相似度（历史记录，外部仓不在本工作区，本次未重跑）：2026-10-21 测得
`caption_snapshots.py` ↔ 目标仓 `caption_snapshot.py` token 相似度 **0.186**，最长连续相同 token
片段 **15** 个，最长连续相同行 **2** 行。

连带修复：`tests/test_ui_scale_settings.py` 未隔离本机外部设置文件覆盖
（`.anima-webui-settings.toml`），在本机必定失败；已补 `_explicit_settings_file_override` 隔离。
这是既有测试隔离问题，非本次改动引入。

交付门槛注记：Python 字节码会占满交付审计的 200 路径上限，因此本轮验证统一使用
`PYTHONDONTWRITEBYTECODE=1`，并在复核前后确认 `find web tests -name '*.pyc' | wc -l` 为 `0`。

## 5. 审计、修复与复核记录（2026-10-21）

「已修 / 已复核」= 修复落地且被独立复核复现确认（当前 revision）；「部分修复」「遗留」见 §6，不计为已修。

| id | 等级 | 最终状态 | 修复要点 / 依据 |
|---|---|---|---|
| BAC-1 | blocker | 已修 / 已复核 | 候选冻结 + commit/rerun/update_items 共用 `_job_locks`；写回期间 rerun → 409 |
| BAC-2 | high | 已修 / 已复核 | restore 先 `_scope_directory` + `_restore_plan` 全量校验，再动盘；非法条目整单拒绝 |
| BAC-3 | medium | 已修 / 已复核 | 解析后无 tag 的候选计 `failed`，不调用 `write_caption` |
| BAC-4 | medium | 已修 / 已复核 | `write_caption` 对 `captions_json` 行、以及 auto + 聚合含本图条目直接拒绝写入：零 sidecar、聚合字节不变；守卫与读取链同源（图片目录 + dataset 根），含嵌套布局；见 §3.1、§6.1 |
| BAC-5 | medium | 已修 / 已复核 | `storage.caption_text_suffix` 单一来源，`write_caption` 与 `jobs.create` 的 skip 预筛同判据；见 §6.1 |
| BAC-6 | medium | 已修 / 已复核 | `apply_trigger` 先拆分 trigger 再 prepend，多标签 trigger 幂等 |
| BAC-7 | low | 部分修复 | 计数已分离（`failed`/`commit_status`）；`conflicts` 仍为死代码，见 §6.2 |
| BAC-8 | low | 部分修复 | `CaptionSnapshotError` 现阻塞且 0 写入；目录解析失败仍写回，见 §6.2 |
| BAC-9 | low | 已修 / 已复核 | 聚合文件名比较改为大小写不敏感 |
| BAC-10 | low | 部分修复 | `_prune` 不再回收持锁 job；无跨任务共享锁，见 §6.2 |
| BAC-11 | low | 已修 / 已复核 | 空 `dataset_file` 时不裁剪，避免跨数据集删除 |
| BAC-12 | low | 遗留 | 既有预览解析问题，非 Phase 1/2 引入，见 §6.2 |
| FAC-1 | high | 已修 / 已复核 | `snapshot`/`snapshot_error` 进入 commit 结果区；无还原点时 `role=alert` 告警 |
| FAC-2 | medium | 遗留 | 快照列表不按 `source` 过滤也不显示来源，见 §6.2 |
| FAC-3 | medium | 已修 / 已复核 | 渲染 `query.error`、提供「重试加载」、备份按钮不再被列表失败禁用 |
| FAC-4 | medium | 已修 / 已复核 | commit 成功后 invalidate 快照列表 |
| FAC-5 | medium | 已修 / 已复核 | `skipped` 只含策略跳过，失败独立计数并使用错误样式 |
| FAC-6 | medium-low | 遗留 | commit 缺「结果未知」对账，见 §6.2 |
| FAC-7 | low | 遗留 | restore 结果无「请重开数据集核对」提示，见 §6.2 |
| FAC-8 | low | 遗留 | `writableIds` 计数与后端判据不一致，见 §6.2 |
| FAC-9 | low | 已修 / 已复核 | `item.error` 与 `commit_error` 分块显示，不再互相吞掉 |
| FAC-10 | low | 遗留 | broken 摘要字段缺失（当前不可达），见 §6.2 |
| FAC-11 | low | 遗留 | 行内按钮可访问名重复、`label` 无 UI 入口，见 §6.2 |
| FAC-12 | low | 部分改善 | 加载中文案已区分；折叠标题计数仍短暂显示 0，见 §6.2 |
| T9-1 | medium | 已修 / 已复核 | 递归数据集嵌套布局下 fail-closed 守卫与读取链目录集合不一致（曾会写 `sub/img.txt`）；t14 对齐目录集合并补 5 条回归，t15 以 11 组嵌套探针复核通过 |

注：t7 的两项收尾已由 round-2 复核（t11，覆盖 t12/t14 之后的当前字节，verdict=pass）与 T9-1 复核
（t15，verdict=pass）确认，状态计入「已修 / 已复核」；t13 的范围限定为 t12 产物（当时 T9-1 仍开放）。

## 6. 已知限制与未修项

本节分两部分：§6.1 是**已按设计决策收尾**的条目（保留记录以防回退，不算缺口）；§6.2 是当前仍未修的
遗留项，每条给出影响与建议的后续处理方式。

### 6.1 已收尾的设计决策（不是缺口）

**BAC-4 [medium，已按 fail-closed 处理（t7 落地、t14 收窄、t15 复核）]：`caption_source_mode=captions_json` 时写回直接拒绝**

- 行为：`write_caption` 在读写任何 sidecar 之前调用 `_assert_sidecar_write_allowed`：行模式解析为
  `captions_json` 时无条件拒绝；`auto` 模式且**读取链会优先读取的** `captions.json`（图片所在目录
  **或** dataset 根目录）已含本图条目时同样拒绝。两种情况都**不新建、不修改任何 sidecar**，聚合文件
  字节不变，错误信息给出两条可操作路径（切换为 sidecar 模式，或直接编辑 `captions.json` 后再打标）。
- 目录集合同源（t14 / T9-1）：守卫用 `_caption_scope_directories(image, row)` 构造
  `[image.parent, row.source_dir, row.image_dir]`（与 `read_caption_for_image` 完全一致），auto 模式直接取
  `read_caption_source_from_dirs` 的 `detected_mode`；因此递归数据集的 `dataset/sub/img.png` + 根
  `captions.json` 也会被拦下，零 sidecar、聚合字节不变。显式 `txt`/`json` 仍早退放行。
- 为什么是设计决策：工作台只写 sidecar，而预览/训练读取链在聚合模式下优先读 `captions.json`；
  继续新建 `<stem>.txt` 会让同一张图出现两份互相看不见的真值。宁可显式失败，也不制造双真值。
- 已知边界（有意保留）：`skip` 预筛只按 sidecar 判据，所以 `captions_json` 行的图片仍会进入候选池，
  要到 commit 阶段才以 `failed` + 中文提示失败；caption 目录零写入（自动还原点仍照常创建），
  只是多一次 provider 调用与一条 `failed` 记录。
- 不要回退：把这里改回「照旧写 sidecar」等于恢复 BAC-4 缺陷。若产品要求直接编辑聚合文件，
  应作为新功能单独评审（需要 `captions.json` 的键级更新与并发策略）。

**BAC-5 [medium，已对齐（t7）]：skip 预筛与写盘共用 `caption_text_suffix`**

- 行为：`storage.caption_text_suffix(settings)` 是后缀的唯一来源，`write_caption` 与
  `jobs.create` 的 skip 预筛都从它取 `text_suffix`；`caption_extension=".tags"` 时预筛能命中、
  `skip` 返回 `skipped=True`、不产生多余的 `.txt`。
- 影响：`append` 基线、`skip` 判据与写盘目标现在指向同一个 sidecar，不再出现「判据看 `.txt`、
  写盘落 `.tags`」。

### 6.2 仍遗留项

**BAC-7 [low，部分修复]：`conflicts` 计数是死代码，item 状态不回退**

- 事实：`CaptionWriteConflict` 全仓无 `raise` 点，`conflicts` 恒为 0、`commit_conflict` 事件永不
  触发；写失败的 item 保持原 state（不回退为 `ready`）。
- 影响：契约字段存在但无实际语义；统计面板永远显示冲突 0。删除字段会破坏响应契约。
- 建议：要么实现真正的冲突检测（写回前比对 `item["caption"]` 与磁盘当前文本），要么在下一版
  契约中显式移除该字段并同步前端类型。

**BAC-8 [low，部分修复]：目录解析失败时仍会写回**

- 事实：`CaptionSnapshotError` 与 `OSError` 现在会中止写回；但数据集行 / 目录解析失败
  （`FileNotFoundError`、非快照类 `ValueError`）仍按「非阻塞」处理，写回继续，只把
  `snapshot_error` 透出到 API 与 UI。
- 影响：与 §2.4「除范围过大外都阻塞」的原始文字仍有差异；用户在 UI 会看到「本次写回没有还原点」，
  但写回已经发生。
- 建议：确认该分级是否符合预期；如需严格，统一改为阻塞并让 UI 直接拦截批量写回。

**BAC-10 [low，部分修复]：无跨任务共享锁**

- 事实：只有 job 级锁；`_prune` 已不再回收仍持锁 job 的锁条目。
- 影响：同一 dataset 的两个 job（或「立即备份」与另一 job 的写回）并发时，快照可能记录到部分
  已写回的中间态。
- 建议：按 `(dataset_file, dataset_index, source)` 引入共享的快照 / 写回锁。

**BAC-12 [low，既有问题]：空 `source_dir` 时写回解析根退化**

- 事实：行配置缺 `source_dir` 却以 `source="source"` 提交时，图片解析根会退化为仓库根，
  而自动快照范围回退到 `image_dir`。
- 影响：理论上存在「写回目标不在还原点覆盖内」的窗口；既有预览解析问题，被 Phase 2 写盘放大，
  非本次引入。
- 建议：单独立项修预览 / 写回共用的路径解析，要求空目录显式报错。

**FAC-2 [medium]：快照列表不按 `source` 隔离**

- 事实：`snapshotApi.ts` 的列表请求与 query key 都不带 `source`，行内也不显示 `source`/`scope_dir`；
  后端列表接口同样不支持按 source 过滤。
- 影响：同一 dataset+index 的「原始图 / 训练图」快照混在一个列表且外观相同，而 restore 实际作用于
  manifest 的 `scope_dir`，用户可能以为自己还原的是当前 source。
- 建议：列表请求 / query key 增加 `source`，后端补过滤参数，行内显示 source（或 scope_dir）。

**FAC-6 [medium-low]：commit 缺少「结果未知」对账**

- 事实：连接中断（status 0）时 `CaptionReview` 没有 `onError` / `needsReconcile`，按钮立即恢复可用。
- 影响：用户可能重复提交破坏性写回；同页其它写操作已有对账入口，行为不一致。
- 建议：与同页对齐，进入 needsReconcile 并提示先核对服务器任务状态。

**FAC-7 [low]：还原结果缺少「重开数据集核对」提示**

- 事实：restore 的 `skipped` 现已恒为空（整单拒绝），但仍无提示说明「任务候选是内存快照」。
- 影响：还原后审核页仍显示旧候选文本，用户可能误以为还原失败。
- 建议：restore 成功后提示重新打开数据集刷新磁盘真值。

**FAC-8 [low]：`writableIds` 计数与后端判据不一致**

- 事实：`CaptionReview` 的 `writableIds` 把 `state=committed` 的项也计入（只判文本非空，不比较是否已写回
  同一文本），而后端会过滤掉「已写回且文本未变」的项。
- 影响：按钮显示「写回全部候选 (N)」，实际可能写出 0 项。
- 建议：前端复用与后端一致的判据。

**FAC-10 [low，当前不可达]：broken 摘要字段缺失**

- 事实：`_broken_entry` 只返回 `id`/`broken`/`created_at_ms`/`totals`，无 `file_count`/`bytes`/
  `created_at`/`reason`/`dataset_file`；因列表按 `dataset_file` 过滤，broken 条目目前不会出现在
  面板里。
- 影响：一旦过滤条件放开，界面会渲染「undefined 个文件」并把 reason 误标为「手动备份」。
- 建议：broken 摘要补齐字段，或前端对缺失字段做兜底。

**FAC-11 [low，a11y]：还原点行的可访问名与标识不足**

- 事实：多行「还原」/「删除」按钮的可访问名完全相同；broken 行按钮禁用但无原因说明；
  `label` 字段 API 支持但 UI 无入口。
- 影响：屏幕阅读器无法区分目标行；手动备份之间除时间外无法区分。
- 建议：补 `aria-label`（含快照 id / 时间）、禁用原因说明与 `label` 输入。

**FAC-12 [low，部分改善]：加载期计数闪 0**

- 事实：列表区在 `query.isPending` 时已显示「正在加载还原点…」，但折叠标题的「还原点（N）」
  仍会在首次加载时显示 0。
- 影响：纯观感问题，无功能影响。
- 建议：加载态隐藏计数或显示占位符。

## 7. 文档修订记录（2026-10-21）

### 7.1 第一次修订（t4 独立复核后）

- 修正 §2.3 第 3 步：旧文字为「目标越界时跳过并计入 `skipped`」；实现改为整单校验 + 整单拒绝，
  `skipped` 恒为 `[]`。同步删除「部分还原」的表述。
- 修正 §2.5：补 `POST /caption-snapshots` 成功状态 **201**；错误映射按实现列出
  （KeyError→404、ValueError/OSError→400、RuntimeError→409、未初始化→503）。
- 修正 §3.2 被吸收键列表：补 `nl`（作为结构性字段先移除、再按条件写回）。
- 新增 §3.3 计数语义：`skipped` 收窄为「按策略跳过」，新增 `failed` 与 `commit_status`；
  说明这是 BAC-7/FAC-5 修正，前端已同步。
- 新增 §2.4 并发与锁、候选冻结（BAC-1）与 §2.3 的整单校验流程（BAC-2）。
- 新增 §5 审计记录与 §6 已知限制与未修项。
- 修正 §4 证据数字：定向 104（旧记录 89）、打标域 164（旧口径 154）、前端 19 文件 / 49 用例（旧 46）。

### 7.2 第二次修订（t7 收尾落地后）

- §3.1 新增「聚合来源 fail-closed（BAC-4）」矩阵：`captions_json` 行与 auto + 聚合含本图条目拒绝写入，
  显式 `txt`/`json` 与 auto 不含条目照常写 sidecar；删掉「预筛尚未对齐」的旧表述，改为
  `caption_text_suffix` 单一来源（BAC-5 已对齐）。
- §3.3 补充：`AggregateCaptionWriteError` 计入 `failed`（不是 `skipped`）；`captions_json` 行的图片
  不会被 skip 预筛掉，最终以 `failed` 呈现。
- §4 证据改为 t7 之后实测：定向 **109**、打标域 **169**、前端 19 文件 / 49 用例、tsc 无诊断、
  ruff 全绿、依赖零 diff；并注明 t7 两项收尾尚无独立复核。
- §5：BAC-4 状态由「遗留」改为「已修（t7，fail-closed）」，BAC-5 由「部分修复」改为「已修（t7）」，
  当时因尚无独立复核而在表下补了一条复核范围说明。
- §6：拆成 §6.1「已收尾的设计决策（不是缺口）」与 §6.2「仍遗留项」，BAC-4 / BAC-5 移入 §6.1 并
  改写为 fail-closed / 已对齐的行为描述（明确「不要当成 bug 改回去」）。

### 7.3 第三次修订（round-2 复核收口后）

- §2.2 之后新增 **§2.2.1 已知限制：快照只覆盖 scope_dir 顶层**（t12 产出，由
  `test_snapshot_scope_stays_top_level_for_recursive_dataset_layouts` 固定）；§2.3 的不变量段补一句指向 §2.2.1。
- §3.1 fail-closed 矩阵：auto 行的聚合查找范围改为「读取链会优先读取的 `captions.json`
  （图片所在目录 或 dataset 根目录）」，并补守卫目录集合同源（`_caption_scope_directories` +
  `detected_mode`）与嵌套布局的说明——t14 实现、t15 复核。
- §4 证据改为 round-2 后的当前 revision：95 / 115 / 171（+4 = 175）/ 19 files · 49 tests / tsc 无输出 /
  ruff 全绿 / 依赖零 diff；三条 `verdict=pass`（t11 / t13 / t15）替换「t7 尚无独立复核」的旧说法；
  补交付门槛注记（`PYTHONDONTWRITEBYTECODE=1`、`*.pyc` 计数为 0）。
- §5：BAC-4 / BAC-5 升为「已修 / 已复核」，删除只表示「尚无独立复核」的中间档图例；新增 T9-1 行；
  表下注改为三条复核的覆盖范围说明。
- §6.1 BAC-4 改写为「t7 落地、t14 收窄、t15 复核」，补充目录集合同源与嵌套布局行为。
- 索引 `docs/findings/README.md` 描述同步为「审计→修复→独立复核闭环 + 已知限制」。
