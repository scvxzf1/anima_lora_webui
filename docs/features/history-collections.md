# 历史任务与集合

状态：稳定
适用版本：当前 WebUI Next 工作台 `/next/history`
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

- `web/frontend-next/src/features/training-history/`（Next 历史任务与集合工作区）
- `web/services/training/` 历史服务
- `configs/web-training-history/`（或外置配置根下的同名目录）
- `tests/test_training_history_*.py`、`tests/test_training_frontend_history.py`

---

## 1. 这是干什么的

一句话：回看已经跑过的训练/预处理任务，按集合整理，并做归档、删除、续训入口。

历史能力包括：

- 最近训练摘要
- 历史任务管理台：搜索、筛选、排序
- 集合管理（collection / collections）
- 批量归档 / 取消归档 / 设置集合 / 彻底删除
- 任务详情：概览、分析、样张与权重（样张卡片可点开查看生成参数与提示词）、日志、配置快照

当前历史模式只保留 **collection / collections**，不再使用旧的 config/flat 主模式。

### 概览「实时指标」网格

训练任务详情「概览」页的实时指标网格固定为 **6 列**，按从左到右、从上到下自然换行；当前共 16 格，排成约 3 行（末行留 2 个空位）。后 6 个配置摘要格来自任务的 `config.snapshot.toml`（API 字段 `config_toml`），不是运行中实时采样：

| 格子 | 含义 | 取值 |
| --- | --- | --- |
| 训练精度 | 训练精度倾向 / 混合精度 | 优先 `precision_preference`，否则原始 `mixed_precision` |
| 训练变体 | 方法短名 | 由 snapshot 标志推断，如 `lora` / `lokr` / `loha` / `vera` / `glora` / `hydralora` / `tlora` / `reft` / `chimera`；识别不出时为 `-`，不回退导入配置文件名 |
| 预处理精度 | 预处理缓存精度 | `preprocess_precision_preference` |
| 块交换精度 | 块交换传输精度 | `block_swap_transfer_dtype` |
| 底模计算路径 | base 前向计算路径 | `base_compute`（如 `bf16` / `w8a16_convrot` / `w8a8_convrot`）；缺省显示 `-` |
| 精度倾向 | 表单「精度倾向」语义 | **不是** snapshot 直读字段：WebUI 保存时会把 `precision_preference` 展开成 `mixed_precision` 并删除；概览按 `precisionPreferenceFromConfig` 规则反推：`mixed_precision=no` → `fp32`，`fp16`/`full_fp16` → `fp16`，否则 `bf16` |

相关代码：`web/frontend-next/src/features/training-history/HistoryOverview.tsx`、`HistorySnapshot.tsx` 及后端 `web/services/training/history_config_chips.py`。

---

## 2. 入口

1. 打开 Next 顶部导航 **历史任务**，进入 `/next/history`。
2. 或从「最近训练」点 **查看全部**。
3. 需要整理集合时点 **集合管理**。
4. 点某个任务进入详情，可看 loss、日志、样张、配置与续训入口。

### 详情头部与产物

- 状态徽章紧贴任务标题；完成使用绿点，与右侧动作按钮区分。训练任务可通过「以此配置新建训练」打开实际配置快照，在命名并确认创建后进入训练页；新文件不会覆盖原配置，也不会自动启动训练。缺少快照时禁用此操作。创建前检查旧输出、缓存路径与续训参数，进入训练页后核对硬件预设。
- 「产物」依次显示训练样张、权重、结果文件。样张默认最新文件优先，步数覆盖在缩略图左上角；切换「Prompt / Seed 演进」后，仅对当前分页按提示词、Seed 和编辑参考图分组，组内按步数从早到晚查看。
- 权重列表只显示文件名、步数、大小、精确步数对应的训练 Loss 与 `FINAL MODEL` 标记；有公共绝对父目录时在标题下统一展示并可复制。公共目录范围为本页，完整文件路径仍可悬停查看或逐项复制，下载与应用到训练草稿保留。
- Loss 来自详情已读取的指标，不匹配验证记录、不用邻近步数估算；缺少对应步数或权重元数据时显示「未记录」。
- 日志、TOML 等结果文件以响应式紧凑卡片排列，宽屏四列；不可用文件保留原因。「配置」页仅显示可搜索、复制的快照，不再重复结果文件列表。
- 「指标」页将 Loss/学习率和显存图表放入独立浅色卡片；GPU 同一采样点选择器下方显示四项紧凑 KPI（显存占用、总显存、利用率、温度）及已读取峰值。概览中的键名左对齐，最新样张摘要附最多四张微型缩略图。

常用筛选：

- 类型：训练 / 预处理
- 状态：完成 / 运行中 / 异常 / 已中断
- 归档：未归档 / 全部 / 已归档
- 来源：队列 / 续训 / 权重热启动
- 训练变体：`lora` / `lokr` / …
- 预处理精度：`bf16` / `fp16` / `fp32`
- 块交换精度：`bf16` / `fp8_e4m3`
- 底模计算路径：`bf16` / `w8a16_convrot` / `w8a8_convrot`
- 精度倾向：`bf16` / `fp16` / `fp32`（由 snapshot 的 `mixed_precision` 等反推）
- 排序：最新、最早、Loss 点数、日志行数、名称

搜索支持任务名、配置名、目录，以及如 `组:骨女`、`配置:lora` 这类快捷写法。

---

## 3. 关键配置项

| 项目 | 说明 |
| --- | --- |
| 历史根目录 | 默认 `configs/web-training-history/`，可随 `configs_root` 外置 |
| 集合 | 用 collection 把相关任务归到一组，便于批量管理和对比 |
| 归档 | 从默认“未归档”列表隐藏，但通常仍保留数据 |
| 任务详情 | 配置快照、路径、权重、样张、日志 |
| 续训 / 热启动入口 | 从历史详情回到配置页续跑 |
| 输出根边界 | 删除运行产物时必须落在全局 `output_root` 允许范围内 |

---

## 4. 危险项

- **彻底删除**：会删历史记录，并可能清理关联运行目录；不可轻易点批量删除。
- **归档误解**：归档不是删除，但会从默认列表消失。
- **批量设置集合**：选错任务会把不相关 run 归到同一集合。
- **从历史续训**：要确认 checkpoint 与配置快照是否匹配当前模型路径。
- **外置配置根切换后**：历史列表会跟随新的 `configs_root`；切根前确认你看的是哪套数据。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_history_list.py \
  tests/test_training_history_delete.py \
  tests/test_training_history_artifacts.py \
  tests/test_training_frontend_history.py \
  -q
```
