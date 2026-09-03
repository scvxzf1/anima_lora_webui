# 数据集编辑器

状态：稳定
适用版本：当前 WebUI 主界面
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：

- `web/static/index.html`（`data-tab="datasets"` / `#tab-datasets`）
- `web/services/config/` 中的 dataset 预设读写
- `tests/test_web_config_datasets.py`

---

## 1. 这是干什么的

一句话：在「数据集」页管理 `configs/datasets/` 下可复用的数据集蓝图，再给训练配置引用。

你可以：

- 新建 / 复制 / 重命名 / 删除数据集预设
- 分组整理、导入导出
- 编辑每个 subset 的路径、caption、缓存、正则化等
- 预览原始图片与 caption

---

## 2. 入口

1. 打开顶部导航 **数据集**。
2. 左侧列表选择一个数据集预设。
3. 右侧编辑器修改字段。
4. 点 **保存**。
5. 回到 **配置** 页，让训练配置引用该数据集预设，或按表单里的数据集选择器绑定。

常用按钮：

- **新建预设 / 复制 / 重命名 / 新建分组**
- **导入 / 导出**
- **删除**
- **刷新**

---

## 3. 关键配置项

| 项目 | 说明 |
| --- | --- |
| 预设文件 | 落在配置根目录下的 `datasets/` |
| 图片路径 / 源路径 | 训练图片所在目录；同名 `.txt` 常作 caption |
| caption 偏好 | 如 prefer json caption 等 caption 读取策略 |
| cache / training 相关目录 | 可按 subset 保留独立缓存路径 |
| regularization | 正则化数据行与权重 |
| 校验划分 | validation split；为 0 时相当于关闭校验 |
| 图片预览 | 读取所选行的图片与 caption，只读预览 |

数据集页改的是“数据蓝图”，不是立刻开始训练。训练真正使用哪份数据，以配置页绑定结果和启动时 runtime 冻结为准。

普通数据行和正则化数据行可以保留各自的分辨率、batch 和缓存设置；训练加载时会在构建 bucket 前按全组普通图片 repeats 配平正则化图片。正则化数据暂不支持分阶段调度，保存时会显式拒绝该组合。

### 通用规则与分组设置

Dragon 数据集页顶部的“通用规则”是新建数据集组的默认基线。修改基线不会
隐式覆盖已有组；只有点击“同步到所有组”后，现有组的对应设置才会更新。

保存预设时，WebUI 会将这些基线值独立写入
`general.custom_attributes.webui_dataset_defaults`。该表只用于恢复 WebUI 表单，
不会替代每个 `[[datasets]]` 中实际生效的分组设置。旧文件没有该元数据时，
仍从第一个数据集组推导基线，保持向后兼容。

---

## 4. 危险项

- **删除数据集预设**：会删掉 `configs/datasets/` 里的蓝图文件；已启动任务不会自动回滚，但新任务会找不到该预设。
- **改路径但不重建缓存**：VAE / text / PE sidecar 与路径绑定；路径变了却继续用旧 cache，容易训到脏缓存。
- **导入同名预设**：默认可能拒绝覆盖，避免误覆盖；覆盖前先确认目标文件。
- **系统只读预设**：可能允许另存，不允许直接改源文件。
- **正则化权重 / validation 设置**：填错会改变训练步数估计和真实训练行为。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_web_config_datasets.py \
  tests/test_web_config_estimation.py \
  -q
```
