# 前端配置优化与严格 Debug 推进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改训练语义的前提下，把前端路径展示、cache token、bridge 装配、启动性能做成可长期推进且每步可测的优化路线。

**Architecture:** 搬家型重构：先统一 shared helper 与装配顺序，再逐步把热点 chunk 业务迁入 feature 模块；禁止新增 globalThis 业务总线。

**Tech Stack:** 现有 WebUI ES modules、pytest 源码契约测试、DOM fixture。

**Spec:** `docs/superpowers/specs/2026-07-11-frontend-config-optimization-design.md`

## Global Constraints

- 用户可见文案简体中文；代码标识英文。
- 测试 `timeout 60` + `.venv/bin/python`。
- 不重写框架，不一次删光 chunks。
- 改 JS 必须同步 cache token（或完成本计划 B1 后走单源）。
- 每个 Task：失败测试 → 实现 → 前端域回归 → 记录。

## Auto Iteration

接入：`docs/superpowers/specs/2026-07-11-five-round-auto-iteration-protocol.md`  
日志：`docs/superpowers/plans/2026-07-11-fullstack-auto-iteration-log.md`

---

## Debug Gate

```bash
timeout 60 .venv/bin/python -m pytest   tests/test_training_frontend_modules.py   tests/test_training_frontend_queue.py   tests/test_training_frontend_history.py   tests/test_training_frontend_config_ui.py   tests/test_training_frontend_dom.py -q
```

---

### Task F1: 统一路径 formatter

**Files:**
- Modify: `web/static/js/shared/format.js`
- Modify: history/queue/dataset/stage 渲染点
- Modify: `tests/test_training_frontend_history.py` / `queue.py` / `modules.py`

- [ ] Step 1: 增加 `formatPathLabel(path, {mode, maxLength})` 契约测试
- [ ] Step 2: 实现 mode=`length|basename|parent-basename`
- [ ] Step 3: 替换双份 `compactPathLabel`
- [ ] Step 4: 所有截断节点补 `title=full path`
- [ ] Step 5: 前端域回归 + 提交

### Task F2: cache token 单源

**Files:**
- Create/Modify: token 常量或校验测试
- Modify: 相关 import / style 链测试

- [ ] Step 1: 写/增强 token 一致性用例
- [ ] Step 2: 单源常量/脚本
- [ ] Step 3: 文档说明 JS/CSS 版本策略
- [ ] Step 4: 回归 + 提交

### Task F3: 高频 bridge 装配收敛

**Files:**
- Modify: `anima-app/index.js` 与 history/config/toml bridges
- Modify: 对应 helpers
- Test: modules + 对应 feature 测试

- [ ] Step 1: 选 history-task-actions 或 config-form 一族
- [ ] Step 2: 去掉默认 globalThis 静默成功路径
- [ ] Step 3: configure 顺序测试
- [ ] Step 4: 回归 + 提交

### Task F4: 启动 import 并行分组

**Files:**
- Modify: `web/static/js/features/anima-app/index.js`
- Test: modules graph

- [ ] Step 1: 无依赖 chunk 分组 `Promise.all`
- [ ] Step 2: 保持 bridge configure 先后约束
- [ ] Step 3: 回归 + 提交

### Task F5: history 列表渲染性能

**Files:**
- Modify: history-list feature / 相关 chunk
- Test: history 测试 + 可选 node fixture

- [ ] Step 1: 建立大数据量渲染基准/契约
- [ ] Step 2: 分片或虚拟化（优先最小侵入分片）
- [ ] Step 3: 回归 + 提交

---

## 与五轮迭代的关系

- R3 锁定本计划
- R4 与后端计划合流优先级
- R5 输出可开工顺序：F1 → F2 → F3 → F4 → F5
