# 预览

状态：稳定
适用版本：当前 WebUI 主界面
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：

- `web/static/js/dragon-ui/pages/preview-workspace.js`
- `web/static/js/dragon-ui/pages/history-sample-dialog.js`（历史详情样张参数弹窗）
- `web/static/css/dragon/06b-dragon-history-sample-dialog.css`
- `web/static/index.html`（classic 预览弹窗）
- `web/routes/preview.py`
- `web/services/preview_service.py`
- `web/services/preview/images.py`
- `web/services/image_listing.py`

---

## 1. 这是干什么的

一句话：查看训练中采样图、推理测试图、自定义目录图片，以及对应权重文件。

两套界面共用同一个预览 API 和路径设置：

- Dragon UI：**模型与系统 → 预览工作区**
- classic UI：训练页 **当前预览**、历史详情里的样张与权重

三种来源：

1. **训练中采样**
2. **推理预览**
3. **自定义路径**

---

## 2. 入口

1. Dragon UI 打开 **模型与系统 → 预览工作区**；classic UI 打开训练页并点 **当前预览**。
2. 在侧边选择来源：
   - 训练中采样
   - 推理预览
   - 自定义路径
3. 训练来源下可选 **训练任务**。
4. 点 **刷新预览图**。
5. 需要时展开 **路径设置**，改目录后 **保存路径设置**。

权重区：

- **刷新权重**
- **正序 / 倒序**
- 查看某轮次、步数对应的权重文件

历史详情（Dragon UI）的 **训练产物 → 样张与权重** 里，点击任意样张卡片会打开「生成参数」弹窗，展示该样张的提示词、负向提示词、原始提示词、分辨率、采样步数、引导系数、种子、采样器、生成时间与提示词文件，并可直接复制提示词、打开或下载原图。弹窗数据来自已加载的 `/api/preview/images` 返回，不发起额外请求。

---

## 3. 关键配置项

| 项目 | 默认/常见值 | 说明 |
| --- | --- | --- |
| 训练任务选择 | 当前任务 / 最新运行目录 | 决定读哪次 run 的 sample 与权重 |
| 兼容训练样张目录 | `output/ckpt/sample` | 旧布局兼容路径 |
| 推理预览目录 | `output/tests` | 生图测试结果常放这里 |
| 自定义目录 | 用户指定 | 支持项目内路径，也支持已保存的绝对目录 |
| 全局输出根 | `output/runs` | 训练 run 与 sample 的主边界 |
| 路径设置持久化 | `configs/web-ui-settings.toml` | 预览路径设置与全局设置分区共存 |
| 单次图片列表 | 最多 500 张 | 返回最近图片；总数仍统计当前筛选下的有效候选 |

图片通常按修改时间展示最新结果；大目录使用 bounded Top-K 扫描，只对最终显示的图片读取尺寸和 PNG metadata。扫描中消失或无法读取的文件会被跳过，损坏图片仍可列出但尺寸显示为空。权重列表可按轮次/步数排序。

---

## 4. 危险项

- **自定义目录填错**：`inference_dir` / `custom_dir` 可以保存绝对目录；读取仍只允许当前任务、全局输出根或已保存的预览目录。
- **删除预览图是永久操作**：不会进入回收站。只允许删除当前所选来源目录的直接普通图片文件；嵌套路径、目录、非图片和 symlink 都会被拦截。
- **批量删除上限**：单次最多 500 个去重后的目标。混合请求会分别返回已删除、已不存在和未删除项，不会因一个越界项掩盖其他结果。
- **选中没有 sample 的历史任务**：不会偷偷回退到“最新 run”，避免看错图。
- **下载权重**：只允许下载受控输出目录内的权重文件。
- **把预览目录指到大盘根路径**：扫描内存已受限，但仍要遍历目录项；不要把磁盘根目录当作图库。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_image_listing.py \
  tests/test_preview_service.py \
  tests/test_cross_domain_delete_boundaries.py \
  tests/test_web_http_contracts.py \
  tests/test_image_test_service.py \
  tests/test_web_route_registry.py \
  tests/test_training_frontend_modules.py \
  tests/test_training_frontend_history.py \
  -q
```

界面模式与 classic 回退见 [Dragon UI 与 classic 兼容界面](dragon-ui.md)。
