# 配置工作台

状态：稳定
适用版本：当前 WebUI 主界面
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：

- `web/static/index.html`（`data-tab="config"` / `#tab-config`）
- `web/services/config_service.py`、`web/services/config/`
- `web/services/training_service.py`
- `tests/test_web_config_service.py`、`tests/test_training_frontend_config_ui.py`

---

## 1. 这是干什么的

一句话：在「配置」页管理训练 TOML，改参数后可直接开始训练或加入队列。

配置工作台覆盖：

- 左侧：项目配置 / 训练输出配置列表
- 中间：表单编辑或直接编辑 TOML
- 顶部操作：加载、保存、另存、删除
- 启动区：GPU 白名单、运行覆盖预设、开始训练、加入队列
- 续接区：从零训练 / 完整续训 / 权重热启动

---

## 2. 入口

1. 启动 WebUI。
2. 打开顶部导航 **配置**。
3. 左侧选中一个配置文件。
4. 需要时点 **加载选中配置**，或在「更多操作」里切换。
5. 改完后点 **保存更新当前选中配置**。
6. 确认 GPU 与训练来源后：
   - **开始训练**：立即启动
   - **加入队列**：进入训练队列等待

目录快捷条（配置目录）可跳到常用类别，例如预览采样相关字段。

---

## 3. 关键配置项

| 区域 | 你在界面上看到的 | 实际作用 |
| --- | --- | --- |
| 配置模式 | 项目配置 / 训练输出配置 | 项目配置改可复用 TOML；输出配置查看历史运行快照 |
| 运行覆盖预设 | `preset-select` | 来自 `configs/presets.toml`，启动时覆盖硬件/采样/性能参数 |
| GPU 选择 | GPU 白名单 | 保存在本机浏览器，启动训练时限制可用 GPU |
| 训练来源 | 从零 / 完整续训 / 权重热启动 | 决定是否恢复状态，或只加载网络权重 |
| 直接编辑配置文件 | TOML 原文编辑 | 跳过表单，直接改文件内容后保存 |
| 开始训练 / 加入队列 | 启动动作 | 会基于当前配置冻结一份 runtime 配置再跑 |

配置合并链（后台）：

```text
base.toml
  -> presets.toml[<preset>]
  -> methods 或 gui-methods 变体
  -> 当前 Web 配置 / CLI
```

---

## 4. 危险项

- **删除当前配置**：会删掉选中的配置文件，先确认不是系统模板或别人还在用的文件。
- **直接编辑 TOML**：语法错误或路径写错会导致保存失败，或训练预检失败。
- **完整续训 / 权重热启动**：选错 checkpoint 会从错误状态继续，或只热启错误权重。
- **开始训练**：会真正拉起训练进程；不要在未保存改动时误以为“界面上看到的”已经落盘。
- **训练输出配置**：主要是只读快照视角；不要把它当成日常可改项目配置。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_web_config_service.py \
  tests/test_web_config_sample_prompts.py \
  tests/test_web_config_file_groups.py \
  tests/test_web_config_preflight.py \
  tests/test_training_frontend_config_ui.py \
  -q
```

补充：

- 启动 / runtime 冻结：`tests/test_training_queue.py`
- 续训选项：`tests/test_training_resume_options.py`、`tests/test_training_resume_actions.py`
