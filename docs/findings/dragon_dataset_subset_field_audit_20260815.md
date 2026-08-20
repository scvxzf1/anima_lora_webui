# Dragon 数据集 subset 字段对照审计

日期：2026-08-15<br>
状态：字段契约已对照，新前端实现中

## 1. 审计口径

本文逐键对照 Classic、现有 vanilla Dragon、React/TypeScript 新前端和
aiohttp 后端 TOML 序列化。Classic/Dragon 只作为交互和文案基线；新前端
不导入旧 DOM、state bridge 或 renderer。

权威后端入口：

- `web/services/config/metadata.py::DATASET_SETTING_KEYS`
- `web/services/config/dataset_rows.py::_normalize_dataset_defaults`
- `web/services/config/dataset_rows.py::_normalize_dataset_rows`
- `web/services/config/dataset_rows.py::_build_dataset_config_doc`

## 2. 全局 defaults

| 键 | Classic | vanilla Dragon | React 新前端 | TOML 位置 / 语义 |
| --- | --- | --- | --- | --- |
| `resolution` | 有 | 有 | 已实现 | `[[datasets]]` 分辨率 |
| `batch_size` | 有 | 有 | 已实现 | `[[datasets]]` 批大小 |
| `prior_loss_weight` | 有 | 有 | 已实现 | 正则数据损失权重 |
| `enable_bucket` | 有 | 有 | 已实现 | 分桶开关 |
| `min_bucket_reso` | 有 | 有 | 已实现 | 最小桶尺寸 |
| `max_bucket_reso` | 有 | 有 | 已实现 | 不能小于 `resolution` |
| `bucket_reso_steps` | 有 | 有 | 已实现 | 桶尺寸步长 |
| `bucket_no_upscale` | 有 | 有 | 已实现 | 禁止放大小图 |
| `validation_split` | 有 | 有 | 已实现 | `0..1` 比例 |
| `validation_split_num` | 有 | 有 | 已实现 | 大于 0 时写入 TOML |
| `validation_seed` | 有 | 有 | 已实现 | 非负整数 |
| `caption_extension` | 有 | 有 | 已实现 | `[general]` 及 dataset 覆盖 |
| `keep_tokens` | 全局有 | 全局有 | 已实现 | 仅 `[general]`，不是 subset 覆盖项 |
| `prefer_json_caption` | 有 | 有 | 已实现 | 兼容键；`caption_source_mode` 为主语义 |
| `caption_source_mode` | 四模式 | 四模式 | 已实现 | `auto/txt/json/captions_json` |

## 3. subset 行字段

| 键 | Classic | vanilla Dragon | React 新前端 | TOML 位置 / 语义 |
| --- | --- | --- | --- | --- |
| `source_dir` | 有 | 有 | 已实现 | `custom_attributes.source_dir` |
| `image_dir` | 有 | 有 | 已实现 | `[[datasets.subsets]].image_dir` |
| `cache_dir` | 有 | 有 | 已实现 | `[[datasets.subsets]].cache_dir` |
| `num_repeats` | 有 | 有 | 已实现 | 最小为 1 |
| `is_reg` | 有 | 有 | 已实现 | 仅 `true` 时显式写入；至少一行普通数据 |
| `recursive` | 有 | 有 | 已实现 | 默认 `true`，仅 `false` 时写入 |
| `path_pattern` | 有 | 有 | 已实现 | 默认 `*`，非 `*` 时写入 |

## 4. 每行 settings 覆盖

每行 `settings` 已覆盖 defaults 中除 `keep_tokens` 外的完整训练、分桶、验证和
caption 字段。React 表单在 hydrate 时先合并 defaults，再覆盖行值；保存时仍按行
提交给后端，由 `_build_dataset_config_doc()` 生成每个 `[[datasets]]` 段。

## 5. 实验规则

| 键 | Classic | vanilla Dragon | React 新前端 | 持久化 |
| --- | --- | --- | --- | --- |
| `nl_tag_mix.enabled` | 有 | 有 | 已实现 | 开启时写入 `custom_attributes.nl_tag_mix` |
| `nl_tag_mix.tag_ratio` | 滑块+数字 | 数字 | 数字 | 内部统一为 `0..1`，读取兼容百分数 |
| `trigger_clone.enabled` | 有 | 有 | 已实现 | `custom_attributes.trigger_clone` |
| `trigger_clone.prompt` | 有 | 有 | 已实现 | 开启时必填 |
| `trigger_clone.num_repeats` | 有 | 有 | 已实现 | 最小为 1 |
| 多 subset 生效范围 | 勾选范围后同步编辑 | 未覆盖 | 已实现显式批量应用 | 复制后每行独立持久化 |

## 6. subset 排序

- Classic：HTML5 drag + Pointer/Mouse/Touch fallback，`Alt+ArrowUp/Down` 键盘移动。
- vanilla Dragon：上移/下移按钮和 `Alt+方向键`，无等价 Pointer 拖放。
- React 新前端：dnd-kit Pointer/Keyboard sensor、显式上下移按钮和
  `Alt+ArrowUp/Down`；重排直接操作 RHF field array，保存时以当前数组顺序写入 TOML。

## 7. 验证

- React 前端：34 个测试通过，包含完整字段 payload、bucket/trigger 校验、
  subset 重排和实验规则批量应用。
- 后端：`test_web_config_datasets.py` 与 `test_dataset_preset_stage_import.py`
  共 59 个定向测试通过。
- 真实浏览器：已验证默认/高级字段、subset 键盘拖动启动/取消、
  内存排序后恢复原值；未向用户预设写盘。
