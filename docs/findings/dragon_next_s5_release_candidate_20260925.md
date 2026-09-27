# Dragon Next S5 发布候选与受控热验收记录

状态：**BLOCKED（候选门禁部分完成；非发布验收）**
日期：2026-09-25

## 范围与前置条件

本轮只推进 S5 的无副作用发布候选证据：独立静态根构建、哈希资产、深链刷新、旧 chunk 保留、`previous-index.html`、原子替换和临时根回退。没有切换默认入口，也没有改写 `web/static/dragon-next/`。

S5 的完整退出仍被前置阶段阻塞：S1 完整人工/读屏器矩阵仍为 `NOT RUN`；S4 虽已有独立的隔离 `PASS` 报告，但其真实服务/provider/数据目录联通仍为 `NOT RUN`，不等同部署态验收。因此本记录不能把 S5 或整个 Dragon Next 标记为发布通过。

按计划要求，真实 GPU 短训、续训、训练中停止、外部 provider、真实下载和 TXT 写回均未执行，继续保持 `NOT RUN`。

## 环境与候选身份

- 分支/HEAD：`dev` / `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`。
- 工作树：开始前已有大量未提交修改；本轮只改 S5 构建目标覆盖、候选验证脚本、前端 README、S5 报告和 Findings 索引，不清理或回滚其他改动。
- 候选目录：`/tmp/dragon-next-s5-release-20260925`。
- 证据 JSON：`/tmp/dragon-next-s5-release-20260925/s5-report.json`。
- 候选入口 SHA-256：`6065771301a7eb64574c0ef82fc0d157a55f5dcff9779932625b5dc960e71a7c`。
- 生产静态目录入口在本轮前后仍为 SHA-256 `4979f04cbd54fd25085baeac1a4f5081758cb6306fef34e01b2c6ebc1031035f`，未被候选构建改写。

## 实施内容

| 项目 | 结果 |
| --- | --- |
| 隔离目标覆盖 | `DRAGON_NEXT_DESTINATION` 只改变显式指定时的构建目标；未设置时仍为 `web/static/dragon-next`。 |
| 候选构建 | **PASS**；`pnpm --dir web/frontend-next build` 在临时目录完成，TypeScript 与 Vite 构建均成功。 |
| 哈希资产 | **PASS**；候选包含 69 个资产，入口引用的 3 个资源均存在；无 `.map` source map。 |
| 深链刷新 | **PASS（候选 fixture + aiohttp handler）**；`/next`、`/next/training`、`/next/history/example-task`、`/next/datasets/workspace/preview` 在候选检查和真实 `next_index_handler` probe 中均返回 HTTP 200。 |
| aiohttp 静态资产/越界 | **PASS（只读最小路由应用）**；`/static/dragon-next/index.html` 与 `TrainingWorkspace-Bhkaog1e.js` 均返回 200 且 `Cache-Control: no-cache`；编码路径穿越请求返回 404。 |
| 旧 chunk 保留 | **PASS**；模拟发布时已有 `legacy-s5-sentinel.js` 未被清空。 |
| `previous-index.html` | **PASS**；发布前旧入口被保留，内容未被候选入口覆盖。 |
| 原子替换 | **PASS**；候选入口写入同目录临时文件后通过 `rename` 替换，未留下临时发布文件。 |
| 回退 | **PASS**；从 `previous-index.html` 写入临时文件并原子替换后恢复旧入口。 |

## 执行命令与结果

```bash
rtk test timeout 180 \
  env DRAGON_NEXT_DESTINATION=/tmp/dragon-next-s5-release-20260925 \
  pnpm --dir web/frontend-next build

rtk test timeout 60 \
  env DRAGON_S5_CANDIDATE=/tmp/dragon-next-s5-release-20260925 \
      DRAGON_S5_REPORT=/tmp/dragon-next-s5-release-20260925/s5-report.json \
  node web/frontend-next/scripts/verify-release-candidate.mjs
```

结果：两条命令均退出码 0；候选验证输出 `assetCount=69`、`previousIndex=true`、`atomicReplace=true`、`rollback=true`、`oldChunkRetained=true`。完整资产哈希和深链状态保存在上述 `s5-report.json`。

真实 aiohttp 静态 handler 只读 probe（不启动完整 WebUI、不加载训练/队列服务）使用 `/tmp/dragon-next-s5-aiohttp-static/static` 作为 `STATIC_DIR`，通过 `aiohttp.test_utils.TestServer` 注册 `next_index_handler` 和 `static_handler`：

```text
/next                                      200 text/html; Cache-Control: no-cache
/next/training                             200 text/html; Cache-Control: no-cache
/next/history/example-task                 200 text/html; Cache-Control: no-cache
/next/datasets/workspace/preview           200 text/html; Cache-Control: no-cache
/static/dragon-next/index.html             200 text/html; Cache-Control: no-cache
/static/dragon-next/assets/TrainingWorkspace-Bhkaog1e.js
                                             200 text/javascript; Cache-Control: no-cache
/static/dragon-next/assets/%2e%2e/%2e%2e/server.py
                                             404 text/plain
```

该 probe 证明候选包经过项目实际 aiohttp 静态 handler 时的深链入口、资产读取和越界拒绝契约；它仍是最小静态路由应用，不代表完整服务 API、HEAD/并发发布或真实生产切换验收。

后端 HTTP/路径安全回归沿用临时 fixture 执行，不指向用户配置根：

```bash
rtk test timeout 180 .venv/bin/python -m pytest -q \
  tests/test_web_static_server.py tests/test_web_http_contracts.py
```

结果：静态服务子集 **27 passed**；HTTP 契约 **48 passed / 1 failed**。唯一失败为 `tests/test_web_http_contracts.py::test_http_config_raw_envelope`，测试仍 monkeypatch 已从 `web.routes.config` 移除的 `load_raw_file`，属于现有配置路由/阶段范围，本轮不改动。

前端工程门禁结果：`pnpm --dir web/frontend-next typecheck` **PASS**；全量 Vitest 当前为 **86 suites passed / 221 tests passed / 0 failed**（JSON 汇总：`/tmp/dragon-next-s5-vitest-20260925.json`）。此前训练配置、训练历史和数据集组件的 5 个失败已由并行修复消除；本轮不改动这些阶段代码。

## 门禁判断

| 门禁 | 状态 | 说明 |
| --- | --- | --- |
| 候选构建与证据快照一一对应 | **PASS** | 候选目录、入口哈希、资产清单和 JSON 报告固定在同一临时根。 |
| G1 生产只读基线 | **PASS（S0 scoped）** | 由 S0 报告提供；本轮不重复执行真实用户服务截图。 |
| 静态服务路径安全 | **PASS（静态测试 + aiohttp probe）** | `tests/test_web_static_server.py`：27 passed；候选 aiohttp probe 的深链/资产为 200，编码路径穿越为 404。 |
| HTTP 契约与前端全量回归 | **BLOCKED** | 静态服务 + HTTP 契约合计 48 passed / 1 failed（HTTP 契约子集 21 passed / 1 failed）；Vitest 86 suites / 221 tests 全通过。唯一 HTTP 失败仍属于既有配置路由测试契约，本轮未越界修复。 |
| S1/S4 前置阶段 | **BLOCKED** | S1 仍 `NOT RUN`；S4 为 scoped `PASS`，但部署态联通仍 `NOT RUN`。 |
| P0/P1 全局开放项 | **NOT RUN** | 本轮只做发布候选文件门禁，不能替代 S1/S4 全量审计。 |
| 默认入口切换 | **NOT RUN** | 未获用户批准，不切换 `/` 或旧 UI 入口。 |
| 真实 GPU/provider/下载/TXT 写回 | **NOT RUN** | 需要用户另行授权，不能由隔离候选通过推断。 |

## 回退与后续

本轮所有发布演练均在 `/tmp/dragon-next-s5-*` 下完成；没有写入生产静态目录、配置、队列、历史、输出或模型。候选验证脚本位于 `web/frontend-next/scripts/verify-release-candidate.mjs`，构建目标覆盖位于 `web/frontend-next/scripts/build.mjs`。

S5 仍不能签收为发布通过。完成 S1、S4 部署态联通和后端门禁后，需在明确批准下再做独立服务热验收和发布切换；默认入口必须保留 `/?ui=dragon`、`/?ui=classic` 回退路径。
