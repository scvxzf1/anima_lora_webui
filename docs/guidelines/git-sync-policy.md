# Git 同步规则

一句话：以后本地 `main` 只和线上 `webui/main` 对齐，避免再混用 `private/main`。

日期：2026-07-05

当前确认：

- 本地分支：`main`
- 跟踪分支：`webui/main`
- 线上远程：`webui`
- 线上地址：`git@github.com:scvxzf1/anima_lora_webui.git`

---

## ✅ 1. 默认目标

一句话：所有普通拉取和推送都只认 `webui/main`。

| 场景 | 默认目标 | 命令 |
|---|---|---|
| 拉取线上更新 | `webui/main` | `git fetch webui --prune` 后按差异决定合并方式 |
| 推送更新到线上 | `webui/main` | `git push webui main:main` |
| 以线上为准同步本地 | `webui/main` | 先比较 `HEAD...webui/main`，需要重置时再确认 |

当前本地 `main` 应跟踪：

```bash
git branch --set-upstream-to=webui/main main
```

---

## 🚫 2. 不再默认使用的远程

一句话：`private` 和 `origin` 都不能再被当成默认线上 main。

| 远程 | 用途 | 默认动作 |
|---|---|---|
| `webui` | 发布目标和唯一同步目标 | 可以 pull / push |
| `private` | 个人主仓或历史镜像 | 不默认 pull / push / reset |
| `origin` | 上游参考仓 | 默认只读，只做审计和选择性合入 |

如果用户没有明确点名 `private`，不要执行：

```bash
git push private main:main
git pull private main
git reset --hard private/main
```

---

## 🔍 3. 推送前检查

一句话：推送前先确认本地只领先 `webui/main`，不要误伤其它远程。

推荐检查：

```bash
git status --short --branch
git fetch webui --prune
git rev-list --left-right --count HEAD...webui/main
git log --oneline webui/main..HEAD
```

如果结果显示远端也有本地没有的提交，不要强推；先合并或单独说明风险。

---

## ⚠️ 4. 高风险同步

一句话：会丢本地内容的操作必须先让用户确认。

以下操作仍然属于高风险：

- `git reset --hard webui/main`
- `git clean -fd`
- 删除未跟踪目录
- `git push --force` 或 `git push --force-with-lease`

执行前必须说明影响范围，并拿到明确确认。
