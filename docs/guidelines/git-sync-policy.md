# Git 同步规则

状态：稳定

默认目标：`github.com/scvxzf1/anima_lora_webui` 的 `main`

本规则按目标仓库 URL 和分支识别线上主线，不要求本机 remote 必须叫 `webui` 或 `origin`。

## 同步前

```bash
git status --short --branch
git remote -v
```

从输出中找到 URL 匹配默认目标的 remote，以下记作 `<target-remote>`。个人 fork、私有镜像和上游参考仓都不能替代默认目标。

```bash
git fetch <target-remote> --prune
git rev-list --left-right --count HEAD...<target-remote>/main
git log --oneline <target-remote>/main..HEAD
```

若目标分支也有本地没有的提交，先说明差异并选择正常合并方式，不要强推覆盖。

## 发布

有目标仓写权限时，明确要求发布后才执行：

```bash
git push <target-remote> main:main
```

没有写权限时，把功能分支推到个人 fork 并向默认目标创建 PR。个人 fork 仍是 fork，不因本地 remote 名称而成为线上主仓。

## 安全边界

- 未跟踪文件默认不提交；推送前检查实际 staged 内容和相关测试。
- `private`、个人 fork 和上游参考仓只有在用户明确点名时才执行写操作。
- `git reset --hard`、`git clean -fd`、工作区丢弃和任何 force push 都需先说明影响并取得明确确认。
- 共享分支撤销提交优先 `git revert`；确需改写历史时只用 `--force-with-lease`。
- 完成后汇报目标仓库、remote、分支、提交 hash，以及剩余未提交或未跟踪改动。
