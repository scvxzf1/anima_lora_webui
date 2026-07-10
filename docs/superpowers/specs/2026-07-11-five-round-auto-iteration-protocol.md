# 五轮自动迭代协议

状态：已启用（R1–R5 文档迭代完成）  
适用分支：`docs/backend-config-optimization`  
关联：

- 后端设计/计划：`2026-07-11-backend-config-optimization-*.md`
- 前端设计/计划：`2026-07-11-frontend-config-optimization-*.md`
- 迭代日志：`docs/superpowers/plans/2026-07-11-fullstack-auto-iteration-log.md`

---

## 1. 每轮输入 / 输出

| 方向 | 内容 |
|---|---|
| 输入 | 当前分支源码、上轮评分卡、未关闭 High、计划进度 |
| 输出 | 后端评分卡、前端评分卡、本轮 Top 项、测试门禁、下轮焦点、是否熔断 |

## 2. 规范评分结构

### 后端 100

| 分项 | 满分 |
|---|---:|
| 架构清晰度 | 15 |
| 配置真相一致性 | 20 |
| 路径/安全边界 | 15 |
| 队列/runtime 稳健性 | 15 |
| stage/resume/progress 贯通 | 10 |
| 测试可防御性 | 15 |
| 可配置性/可运维 | 10 |

### 前端 100

| 分项 | 满分 |
|---|---:|
| 模块边界与入口 | 15 |
| 状态管理纯度 | 15 |
| 过渡层控制 | 15 |
| 测试护栏 | 15 |
| 热点文件控制 | 10 |
| 性能 | 10 |
| UX 一致性 | 10 |
| a11y/稳健性 | 10 |

等级：A90+ / B80-89 / C70-79 / D<70

## 3. 子代理分工

| 角色 | 任务 | 权限 |
|---|---|---|
| backend-auditor | 后端健康评分 | 只读 |
| frontend-auditor | 前端优化项与评分 | 只读 |
| planner | 合流写文档 | 写 docs |
| test-auditor | 门禁有效性 | 只读/补测 |

规则：auditor 默认可并行；planner 等结果后收口；`max_depth=1`。

## 4. 严格 Debug 测试门禁

1. 失败测试（红）
2. 最小实现
3. 域包 ≤60s
4. 跨域回归
5. 记命令与结果

### 后端跨域最小回归

```bash
timeout 60 .venv/bin/python -m pytest   tests/test_training_queue.py   tests/test_training_runtime_config_core.py   tests/test_training_history_list.py   tests/test_preview_service.py   tests/test_web_config_preflight.py   tests/test_stage_schedule.py -q
```

### 前端域最小回归

```bash
timeout 60 .venv/bin/python -m pytest   tests/test_training_frontend_modules.py   tests/test_training_frontend_queue.py   tests/test_training_frontend_history.py   tests/test_training_frontend_config_ui.py   tests/test_training_frontend_dom.py -q
```

失败诊断：

1. 状态错误还是路径错误？
2. configs_root / output_root / history / queue / runtime fixture？
3. monkeypatch 是否盖住真实入口？
4. 字符串契约是否噪音？
5. queue backup / orphan / launch lock？
6. cache token / DOM id / bridge configure？

## 5. 五轮目标递进

| 轮次 | 目标 |
|---|---|
| R1 | 基线评分 + 协议锁定 |
| R2 | 后端 High 对照 feat 落地 |
| R3 | 前端计划冻结 |
| R4 | 全栈优先级合流 |
| R5 | 执行队列冻结，可开工 |

## 6. 停止 / 熔断

- 同一 High 连续 2 轮无收敛 → 熔断扩 scope
- 测试门禁无法 60s 给信号 → 先补 smoke
- 写集冲突 → 改串行
- 用户数据删除类 → 显式确认

## 7. 每轮表格模板

```markdown
### Round N — YYYY-MM-DD
| 项 | 值 |
|---|---|
| 分支 | |
| 后端总分/等级 | |
| 前端总分/等级 | |
| 本轮焦点 | |
| 完成项 | |
| 新增 High | |
| 下轮焦点 | |
| 测试门禁 | |
| 熔断? | |
```
