# 配置工作台

状态：稳定（Classic / Dragon 兼容说明；Next 能力单独标注）
适用版本：当前 WebUI 主界面与 `/next/training`
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

本页第 1-5 节以 Classic/Dragon 配置页为主；第 6 节单独说明 Next React `/next/training` 的能力边界。

配置工作台覆盖：

- 左侧：项目配置 / 训练输出配置列表
- 中间：表单编辑或直接编辑 TOML
- 顶部操作：加载、保存、另存、删除
- 启动区：GPU 白名单、运行覆盖预设、开始训练、加入队列
- 训练来源：Classic / Dragon 配置页可选从零训练、完整续训或权重热启动；Next `/next/training` 配置页不提供续训模式选择，完整续训从历史任务详情页发起

---

## 2. 入口

1. 启动 WebUI。
2. 打开顶部导航 **配置**。
3. 左侧选中一个配置文件。
4. 需要时点 **加载选中配置**，或在「更多操作」里切换。
5. 改完后点 **保存更新当前选中配置**。
6. 在 Classic / Dragon 配置页确认 GPU 与训练来源后：
   - **开始训练**：立即启动
   - **加入队列**：进入训练队列等待

目录快捷条（配置目录）可跳到常用类别，例如预览采样相关字段。

---

## 3. 关键配置项

| 区域 | 你在界面上看到的 | 实际作用 |
| --- | --- | --- |
| 配置模式 | 项目配置 / 训练输出配置 | 项目配置改可复用 TOML；输出配置查看历史运行快照 |
| 运行覆盖预设 | `preset-select` | 来自 `configs/presets.toml`，启动时覆盖硬件/采样/性能参数 |
| GPU 选择 | GPU 白名单 | Classic / Dragon 与 Next 均将选择保存在本机浏览器；Classic 在启动/入队时、Next 在预检/启动/入队时发送 `gpu_whitelist` |
| 训练来源 | 从零 / 完整续训 / 权重热启动 | Classic / Dragon 配置页提供三种来源；Next 配置页只有 `network_weights` 等权重字段，完整续训在历史任务详情页发起 |
| 直接编辑配置文件 | TOML 原文编辑 | 跳过表单，直接改文件内容后保存 |
| 开始训练 / 加入队列 | 启动动作 | Classic / Dragon 普通启动要求当前配置先保存并执行服务端预检；完整续训改用历史冻结配置并单独审查来源。非 runtime 配置可能进入预处理，入队会冻结配置并暂停等待。Next 脏草稿先自动保存，再由服务端预检；非 runtime 配置可能进入预处理，入队请求使用 `start_paused=true` |

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
- **完整续训 / 权重热启动**：选错 checkpoint 会从错误状态继续，或只热启错误权重。Classic / Dragon 配置页提供完整续训来源选择；Next 配置页不提供该模式，需从历史任务详情发起。
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

## 6. Next React 配置工作台

新版 `/next/training` 按输入准备、方法配置、训练计划、资源四个配置阶段切换，
每次只渲染当前分类，不再将所有分类串成长滚动表单。切换分类保留未保存草稿，
并从该分类顶部开始浏览。搜索和参数视图筛选覆盖全部分类，筛选时标签显示各分类
匹配数量；当前分类无匹配项时显示空状态。预检是独立操作和结果面板，不是第五个字段阶段；保存、预检和入队仍针对整份配置。

“资源”阶段内部按精度与计算后端、块交换与内存卸载、梯度检查点、编译加速、
预处理与数据加载、诊断折叠分组；有适用字段时另显示多卡与并行。
分组标题显示当前配置摘要，AUTO 开关及其关联参数排在手动块交换数量之前。
搜索或“已修改”筛选自动展开匹配分组，不改变字段值、适用条件或保存范围。

输入准备、方法配置、训练计划也使用同一套折叠分组。输入准备按模型、数据集、
标注与遮罩、筛选、缓存排列；方法配置按基础方法、热启动、容量及适配器分支排列；
训练计划按任务名称、训练量、优化器、损失、正则化、预览、保存、日志排列。
基础组默认展开，其他组按需展开；目录之外的字段保留在“其他配置与审计”，不丢弃。

新版样张提示词支持图形化与原文模式，可在“训练预览”的提示词字段旁或顶部操作栏打开。
图形化模式逐条编辑正负提示词、尺寸、步数、CFG、种子、Flow shift、采样器与额外参数，
支持复制、排序、删除。未改动的行及注释、空行保留原文；被编辑行规范化参数顺序。
未关联配置优先读取同名分叉文件，仅在文件不存在时回退共享默认文件；界面显示关联状态。
点击保存才写入配置专属文件并关联当前配置，不自动覆盖旧文件。文件是否存在的标记需要
更新后的 Web 后端；更新已有服务后需重新启动 WebUI 才能获得该标记。

Dragon Classic 入口为 `?ui=dragon#config/training-config/all`；Next 入口为 `/next/training`。
Next 参数工作台保留输入准备、方法配置、训练计划、资源四个阶段，预检通过独立按钮/结果面板进入；
搜索、配置预设库和保存行为的接口契约仍与后端共享，但页面能力不等同于 Classic 配置库。

Classic Dragon 训练配置预设库使用带文字的胶囊操作：新建、导入、导出、重命名、另存为、新建分组。
新建使用内置 `configs/gui-methods/lora.toml` 空白预设，并写入全局模型默认值；新文件保存到
`configs/imported/`。重命名只修改当前可编辑训练配置的文件名，保留内容、目录和分组排序，
不会覆盖同名文件，也不会改写既有训练历史或队列。另存为复制已保存的当前配置；
存在未保存修改时，先经过页面的修改确认。命名操作统一使用 Dragon 内置弹窗。
Next `/next/training` 当前提供配置库读取/搜索、保存、另存、配置文件重命名、分组新建/重命名/删除、
文件移动和排序，以及 TOML 编辑器中的导入/导出。页面没有删除配置文件的操作；删除分组不会删除其中的配置文件。
Next 另存包含当前草稿，不会覆盖同名文件，另存成功后切换到新文件，不弹出“未保存修改确认”。
定向测试入口为 `tests/test_training_preset_file_actions.py`。

Anima、Krea-2 和 Z-Image 的模型族均可在 Next 中选择，但网络字段和选项按能力目录动态过滤；
选项开放不等于全部组合已验证，未列出的组合由服务端预检拒绝。完整运行边界见
[多模型支持现状](../multi_model_support.md)。完整续训仍在历史任务详情页，不属于 Next 配置页的训练来源选择。

- 浅色与深色共用字号、间距、控件尺寸和状态语义。蓝色用于主要操作，绿色表示已同步，
  琥珀色表示未保存修改，红色表示预检错误；不可用项保留文字原因与帮助入口。
- 宽屏限制参数阅读宽度，窄屏改为单列表单和配置抽屉。帮助、模型与数据集选择弹窗同样适配两个主题。
- 主题入口为 `web/static/css/dragon/04h-dragon-config-visual-system.css`；布局、字段、弹窗分别位于
  同目录的 `config-workbench/structure.css`、`properties.css`、`dialogs.css`，避免叠加长补丁。
- 静态契约测试：`tests/test_dragon_config_visual_system_frontend.py`。真实浏览器验证入口：
  `tests/browser/dragon-config-visual-smoke.cjs`，需要 Node.js、可解析的 `playwright` 包与 Chromium。

浏览器验证需已有选中的训练配置、模型预设，以及含可预览图片的数据集预设。先启动 WebUI，再执行：

```bash
DRAGON_QA_URL=http://127.0.0.1:20102/ node tests/browser/dragon-config-visual-smoke.cjs
```

可通过 `NODE_PATH` 指定 Playwright 依赖目录、`DRAGON_QA_BROWSER` 指定 Chrome 可执行文件。
脚本使用独立浏览器上下文，拦截非只读 API 请求，不保存配置、不启动训练；覆盖两主题、
390 至 2499 像素窗口、80% / 125% 页面缩放、弹窗、侧栏、搜索、修改与撤销。
截图与检查结果默认写入 `/tmp/dragon-config-visual-qa`，可用 `DRAGON_QA_OUTPUT` 更改。

Next React 工作台的定向入口另见 `web/frontend-next/src/features/training-config/`：

```bash
pnpm --dir web/frontend-next exec vitest run src/features/training-config
pnpm --dir web/frontend-next run typecheck
```
