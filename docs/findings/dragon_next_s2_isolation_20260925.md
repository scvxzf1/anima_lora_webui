# Dragon Next S2 隔离写入验收记录（2026-09-25）

状态：**PASS**（仅 Dragon Next S2 隔离写入；非发布验收）
范围：只推进阶段计划 S2；未执行或改动 S0、S1、S3、S4、S5。
基线：`dev`，HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`，相对 `origin/dev` ahead 13。工作区已有大量修改、删除和未跟踪文件；均保留。本轮只修改 S2 的配置写入服务、Next 编辑链路、定向测试、本报告及计划 S2/V-05 状态。

## 隔离边界

- 后端使用 `aiohttp.test_utils.TestClient/TestServer` 注册真实 config/settings routes；每个 pytest 用 `tmp_path` 配置树、settings、训练 TOML、dataset TOML 和 sample-prompts 文件。未连接现场 `20203`，未读写仓库内真实 `configs/`。
- 配置根切换用临时项目根的 `.anima-webui-settings.toml` 和生产 `DynamicPath` 解析；不覆盖本机设置。
- 浏览器用 Playwright/Vite `http://127.0.0.1:5174`、mock API/WS、默认主题与视口。没有启动真实训练、provider、下载或真实配置写入。
- 静态发布包哈希与现场静态包身份：**N/A**，本阶段未使用生产静态包。浏览器测试未保存验收截图；临时 Playwright 失败 trace 不作为通过证据。

## 验收结果

- 训练配置 HTTP 链：预览 diff 不写源文件；PATCH 后 GET 重载与磁盘一致；未知键保留；另存拒绝覆盖；rename 后路径与文件一致。
- 样张文本：注释、空行、首尾空格和换行格式往返不变；越界路径返回 400，未创建 root 外文件。
- 数据集 HTTP 链：多 subset 顺序及 stage schedule 持久化；创建分组、放置 preset、应用到训练 TOML 后，GET 与磁盘内容一致。注入训练配置写入失败时 apply 返回 400，目标 TOML 保持原样。
- 模型库：成功写入后提交旧 revision 返回 409，刷新仍是首次写入。设置根切换后模型库随 settings 文件可读；旧训练/数据集配置留在原根，未搬入新根。
- 设置页：先缓存数据集/模型查询，再切换配置根；pending 时保存控件锁定且设置 PUT 只有一次；回到工作区后数据集和模型均重取新根数据。
- 失败与未知结果：路径错误返回 400；注入写前磁盘错误返回 500 且原文件不变、请求未自动重试；提交前断网保留草稿且不重试；模拟服务端已提交但响应断开时显示“结果尚未确认”、保留草稿且只发一次请求，刷新读回服务端已保存值。
- 双标签冲突：训练 TOML 的 PATCH/PUT、数据集预设 PUT、样张 TXT 的已有文件和不存在的 fork PUT、全局设置 PUT 都携带读取 revision；首次写入成功，旧 revision 返回 409，磁盘保持首次内容。模型库原有 revision 409 继续通过。文件锁覆盖比较与原子替换，另存的不可覆盖检查也移入锁内。
- Next 草稿恢复：训练结构化编辑、原始 TOML、数据集编辑、样张提示词和全局设置在 409 后不自动重试；草稿保留，重复保存被阻止，需用户确认重新读取。提示词先写 TXT、后更新 TOML 引用时，若第二步失败会显示部分成功路径，不宣称事务回滚。

## 验证命令

- `rtk test timeout 180 .venv/bin/python -m pytest -q tests/test_dragon_next_s2_isolation_acceptance.py tests/test_dragon_next_training_config_acceptance.py tests/test_dragon_next_dataset_acceptance.py tests/test_web_config_sample_prompts.py tests/test_web_config_raw_files.py tests/test_web_raw_rename_http.py tests/test_model_config_service.py tests/test_global_settings_runtime.py`：89 passed（最终串行复跑）。
- `rtk test pnpm --dir web/frontend-next typecheck`：通过。
- `rtk test timeout 300 pnpm --dir web/frontend-next exec vitest run --maxWorkers=1 --fileParallelism=false`：49 文件、220 用例通过。默认并发 `web-next-check` 曾出现 5 秒用例超时及受负载影响的断言失败；单独定向复跑 23/23 通过，不能把默认并发门禁记为通过。
- `rtk test pnpm --dir web/frontend-next exec playwright test e2e/s2-isolation.spec.ts --workers=1`：exit 0；根切换/重复提交、提交后断网、设置 409 草稿保留 3 项通过。
- `rtk test pnpm --dir web/frontend-next exec playwright test e2e/state-feedback.spec.ts --grep=model --workers=1`：exit 0；模型冲突交互回归通过。
- `rtk test pnpm --dir web/frontend-next exec playwright test e2e/state-feedback.spec.ts --grep=network --workers=1`：exit 0；提交前断网交互回归通过。
- `rtk proxy .venv/bin/ruff check` 针对本轮修改的 Python 源码与 S2 测试：All checks passed；`rtk git diff --check` 通过。
- `tests/test_documentation_integrity.py`：7 passed、1 failed；S2 报告已加入 findings 索引，剩余失败仅因 `adaptive_runtime_20260921.md`、`adaptive_training_20260922.md` 两份非 S2 文档未被索引，本轮按范围约束未改动。

## 问题与后续

- P0：本轮隔离测试未发现。
- P1：本轮 Next S2 范围内已关闭 V-05。旧客户端省略 revision 时仍可使用兼容写入，直接文件系统写入也不受 WebUI 文件锁约束；不能把此结论扩展为全仓所有写入者的事务保证。
- P2：本轮未登记。
- G1：N/A；S0 生产只读门禁不属于本阶段。默认并发前端门禁和文档索引的上述失败如实保留，不作为 S2 写入契约的通过证据。
- 未执行真实配置根、真实 settings、运行中的训练/队列/历史及外部服务写入；这些不构成通过证据。
- 回退方式：仅回退本轮 S2 的 revision 服务、路由、Next 表单与验收测试/记录；不动已有用户配置或训练数据。没有改动真实运行数据。
