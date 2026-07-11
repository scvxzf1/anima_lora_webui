# 训练队列

状态：稳定
适用版本：当前 WebUI 主界面
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：

- `web/static/index.html`（`data-tab="training"`，队列视图）
- `web/services/training/` 队列与 runtime
- `configs/web-training-queue/`（或外置配置根下的同名目录）
- `tests/test_training_queue.py`、`tests/test_training_frontend_queue.py`

---

## 1. 这是干什么的

一句话：把多个训练任务排成队，按顺序跑，并在失败时决定暂停还是继续。

队列能力包括：

- 从配置页 **加入队列**
- 训练页查看队列摘要与管理台
- 暂停 / 恢复
- 失败策略：暂停队列 或 继续下一个
- 批量取消、清理已完成 / 已取消
- 强制中止当前队列

---

## 2. 入口

1. 在 **配置** 页准备好配置后点 **加入队列**。
2. 打开顶部导航 **训练**。
3. 点 **队列** 视图，或点摘要区的 **管理** 打开队列管理台。
4. 需要时用：
   - **暂停 / 恢复**
   - **失败后** 下拉框
   - 更多操作：中止后续、强制中止、取消全部、清理完成项

筛选：

- 待处理 / 全部 / 等待 / 运行 / 异常 / 完成 / 已取消

---

## 3. 关键配置项

| 项目 | 说明 |
| --- | --- |
| 队列条目 | 入队时会冻结 runtime 配置，之后改原 TOML 不一定影响已排队任务 |
| 失败策略 | `pause`：失败后暂停；`continue`：失败后继续下一个 |
| 暂停队列 | 可只排队、不自动开跑，方便手工确认 |
| GPU 白名单 | 来自配置页浏览器侧选择，影响启动时可用 GPU |
| 输出根目录 | 全局设置 `output_root`，决定运行目录落点 |
| 队列存储 | 默认 `configs/web-training-queue/`，可随配置根外置 |

---

## 4. 危险项

- **停止训练 / 强制中止队列**：会打断正在跑的任务，可能留下不完整 checkpoint。
- **取消全部队列 / 取消全部等待**：批量清掉未完成项，不可靠“撤销”。
- **失败后继续下一个**：某个任务失败时不会停，后续任务会继续烧 GPU。
- **清理已完成 / 已取消**：清的是队列记录，不等于删除历史任务产物；但会丢掉队列侧状态。
- **一边改配置一边等队列**：已入队任务通常使用冻结配置；别以为改完左侧 TOML 就会改已排队任务。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_queue.py \
  tests/test_training_queue_resume.py \
  tests/test_training_frontend_queue.py \
  -q
```
