# Dragon Next S0 只读基线（2026-09-24）

状态：PASS（S0 门禁与版本溯源）；不代表功能发布验收

## 执行环境

- 分支：`dev`；HEAD：`5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`。
- 开始时工作树已有大量修改、删除和未跟踪文件，覆盖配置、文档、训练代码与 WebUI；本轮只新增巡检脚本修改、本记录和索引链接，未清理或覆盖其他改动。
- 服务：`http://127.0.0.1:20203`，本机 aiohttp 返回 HTTP 200。
- 浏览器：Chrome `153.0.8010.36`；视口：1440×900、1280×720、390×844、360×800。
- 真实后端只用于 GET；脚本继续拦截并记录所有非 GET/HEAD/OPTIONS `/api/**` 请求。本轮 `blockedCommands=[]`，未保存设置、写配置、改队列、触发任务或训练。
- 页面数据来自本机服务；无临时 fixture 根。截图保存在本机临时目录，可能含用户配置内容，不纳入仓库或外发。

## 验证结果

- 修复 V-01：训练页的“输出名称”只在“训练计划”阶段挂载。巡检现在先点击该阶段 tab，再检查 textbox，不再把分阶段表单的后续字段误作首屏就绪信号。
- 八个主路由在四种视口下共 32 次直达加载：全部 HTTP 200，H1 与预期一致，document 级横向溢出为 0。
- 页面异常、控制台错误、Dragon Next 静态资源错误均为 0；浏览器收到的 61 个静态资源均为 HTTP 200；未观察到非只读 API 请求。
- 代表训练页主题：默认深色、切换浅色后 `data-theme=light`；缩放通过隔离浏览器中覆写 `GET /api/settings/global` 响应测试，125%、150%、200% 均生效且无 document 级横向溢出。没有向服务发送设置写入。
- 服务端入口 HTML SHA-256：`4979f04cbd54fd25085baeac1a4f5081758cb6306fef34e01b2c6ebc1031035f`；ETag：`"18d840ce832a1234-309"`；最后修改：`Thu, 24 Sep 2026 12:15:08 GMT`。
- 在 `/tmp/dragon-next-s0-build-20260924` 对当前工作树做隔离 Vite 构建：69 个构建资源均存在于发布静态目录，同名资源哈希不一致数为 0；构建入口 HTML 与服务端入口逐字节同哈希。巡检实际加载的 61 个资源与隔离构建也逐项哈希一致。没有运行会发布静态包的 `web-next-build`，没有更改 `web/static/dragon-next/`。
- 首路由 JS gzip 合计：`194641` bytes。
- Dragon Next typecheck 与 Vitest：49 个测试文件、218 个用例全部通过。

## 命令与证据

- `rtk proxy env DRAGON_VERIFY_URL=http://127.0.0.1:20203 DRAGON_VERIFY_OUTPUT=/tmp/dragon-next-production-20260924-s0-r2 node web/frontend-next/scripts/verify-production.mjs`：退出码 0。
- `rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check`：通过。
- 隔离构建：`rtk proxy timeout 180 pnpm --dir web/frontend-next exec vite build --outDir /tmp/dragon-next-s0-build-20260924 --emptyOutDir`：通过，仅写入临时目录。
- 完整 JSON：`/tmp/dragon-next-production-20260924-s0-r2/report.json`。截图同目录，另有 `training-scale-{125,150,200}.png` 与 `training-light-1280x720.png`。截图只供本机核验。

## 风险状态与边界

- G1 / V-01：PASS，生产只读脚本可完整生成八页 × 四视口报告。
- V-02：PASS，本轮工作树构建、发布静态目录与服务端实际加载资产已通过入口与资源哈希关联。
- 产品 P0/P1/P2 mutation、冲突、训练和外部服务场景：NOT RUN；这些不属于 S0 只读门禁，也不能由本报告推断通过。
- 队列、历史、设置、数据集、蒙版和标注的写入仍需后续阶段使用临时根/stub 验证。没有真实启动、停止、入队、删除、TXT 写回、模型下载或计费 API 调用。
- 下一阶段按计划进入 S1：壳、导航、键盘焦点与加载/错误/dirty/busy 等状态语义；S2 及之后尚未验收。
