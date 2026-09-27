# Dragon Next S4 图片、蒙版与打标闭环验证报告

状态：**PASS（仅 S4，非发布验收）**

日期：2026-09-25

分支/HEAD：`dev` / `5b60d111`

范围：Dragon Next 图片工作台、蒙版编辑与打标审核/写回；不改变 S0-S3、S5 的任务范围或状态。

## 结论

本轮在临时目录和本地 provider stub 上验证了图片工作台上下文、返回滚动恢复、蒙版保存与 revision 冲突、目标子集批量应用，以及 caption 候选生成、人工审阅和显式 TXT commit。复核后又使用临时 aiohttp 与当前 Vite 源码通过蒙版浏览器真实 HTTP 探针。覆盖范围满足 S4 的隔离验收目标；这不是部署态或发布验收，不能据此批准切换默认入口。

## 本轮覆盖

- 图片工作台缺少 dataset 上下文时显示恢复路径；图片预览失败与可编辑文本分离，重试后可恢复。由路由、DatasetWorkspace 组件和 mock Playwright 测试共同覆盖。
- DatasetWorkspace 单测在打开工作台前设置两个独立滚动容器的位置，返回后断言两个位置分别恢复。
- 蒙版 API/服务层覆盖图片列表与读取、PNG 保存、revision 冲突、无效目标校验、批量应用持久化和路径保护。批量应用要求显式目标子集；所选子集共享当前蒙版目录，未匹配同名蒙版的图片按无蒙版处理。
- 前端 mock E2E 覆盖蒙版 dirty 离开保护、切图/切子集保护、保存冲突保留草稿并重试、批量 apply pending/冲突/失败恢复。
- captioning 的临时 HTTP 测试使用本地 provider stub，验证图片请求、候选生成、对选中候选的审核修改和显式 TXT commit；未选中的 sidecar 保持原文。测试同时检查 provider 凭据不进入任务响应/日志。
- caption 前端测试覆盖跨页对象身份、候选草稿保存失败后的显式重试、TXT commit 未知结果保持 pending 且不自动重试、切换任务时显式丢弃草稿、翻译候选仅在明确点击后应用，以及日志和图片预览失败恢复。

## 验证结果

| 验证 | 结果 |
| --- | --- |
| `timeout 90 .venv/bin/python -m pytest tests/test_web_mask_editor.py tests/test_tagging_service.py -q` | **38 passed, 1 warning**；aiohttp 测试 app 使用字符串 key 产生 `NotAppKeyWarning` |
| mask/caption Vitest：`maskCanvas.test.ts`、`CaptioningPage.test.tsx`、`CaptionReview.test.tsx` | **3 files / 7 tests passed** |
| 图片工作台 Vitest：`DatasetWorkspace.test.tsx`、`DatasetImageWorkspacePage.test.tsx`、`DatasetMaskEntry.test.tsx` | **3 files / 31 tests passed** |
| 翻译 Vitest：`CaptionTranslation.test.tsx` | **1 file / 1 test passed** |
| `pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts e2e/caption-context.spec.ts --output=/tmp/dragon-s4-e2e-20260925` | **21 passed** |
| `pnpm --dir web/frontend-next exec playwright test e2e/final-audit.spec.ts --grep 'sample image failure|caption logs retry locally' --output=/tmp/dragon-s4-final-audit-20260925` | **2 passed** |
| `pnpm --dir web/frontend-next typecheck` | **通过** |
| `.venv/bin/ruff check web/services/config/mask_editor.py web/routes/mask_editor.py tests/test_web_mask_editor.py tests/test_tagging_service.py` | **通过** |
| `timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q` | **8 passed** |

上述命令在本仓库中均通过 RTK 包装器运行；前端 Playwright 输出留在 `/tmp/dragon-s4-e2e-20260925` 和 `/tmp/dragon-s4-final-audit-20260925`。

## 隔离与未执行项

- Python HTTP 测试使用 pytest 临时目录、临时图片/TXT 和 aiohttp TestClient；caption provider 是本机进程内 stub。没有访问真实 provider、密钥服务或外部网络。
- 定向 Playwright 套件使用 Vite 测试服务器并 mock API/WS；另有单独的真实 HTTP 蒙版探针连接临时 aiohttp。两者均未连接当前生产 WebUI，也没有替换 `web/static/dragon-next/`。
- 未启动训练或下载模型；未修改用户真实图片、蒙版、TXT、数据集配置或 WebUI 设置。
- **NOT RUN**：部署态 Next UI 的完整联通、caption 浏览器到 provider stub 的真实 HTTP 贯通、真实 provider 调用、真实用户目录写回、GPU 训练及发布静态包验收。

## 剩余覆盖边界

定向浏览器 E2E 的接口由 mock 提供，真实蒙版浏览器 HTTP 链路另经临时服务探针覆盖。caption 浏览器到真实 aiohttp 的请求/响应组合仍未贯通。未穷举只读 UI、上传超限和各类 malformed query 参数；profile 与 secret 分别原子写入，跨文件失败不是事务性的，需在后续专项处理。上述缺口不改变已覆盖 S4 隔离闭环的结论，但不能外推为全面 UI 或生产环境验收。

## 复核补充（2026-09-25）

本次审阅发现并修正四处 S4 范围内问题：

1. 蒙版 apply 的非法 JSON 原会退化为旧式空请求，从而在携带有效 `If-Match` 时写入当前子集；现在返回 JSON 400，配置文件字节不变。错误媒体类型和无效查询参数也统一返回 JSON 错误，供 Next API 客户端显示。
2. 本地 WD14/CLTagger profile 原可通过直接 API payload 落盘无意义的 API key；现在创建和更新均在任何 profile 写入前拒绝，路由返回 400，secret 文件不新增该值。
3. 图片工作台的 `returnTo` 若不是合法 `/datasets` 路径，原仍执行历史后退；现在回到经过校验的默认数据集路径。
4. 蒙版页本地切子集原不更新 URL，跨到预览/打标会重新使用旧子集；现在等切图稳定后更新 `subset`，避免 dirty guard 二次拦截。mock E2E 确认脏态确认、URL 与三视图保持子集 2。

| 本次补充验证 | 结果 |
| --- | --- |
| `timeout 150 .venv/bin/python -m pytest tests/test_web_mask_editor.py tests/test_tagging_profiles.py tests/test_tagging_service.py -q` | **51 passed, 2 warnings**；warning 仅为测试 app 字符串 key 的 aiohttp `NotAppKeyWarning` |
| `pnpm --dir web/frontend-next exec vitest run`（7 个 S4 定向文件） | **7 files / 40 tests passed** |
| `pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts e2e/caption-context.spec.ts --output=/tmp/dragon-s4-review-e2e-20260925` | **21 passed**，其中新增切子集后三视图一致性断言 |
| `pnpm --dir web/frontend-next typecheck`；Ruff 定向检查 | **通过** |
| `MASK_HOT_URL=http://127.0.0.1:20544 MASK_HOT_OUTPUT=/tmp/dragon-s4-hot-fresh-20260925 node web/frontend-next/scripts/verify-mask-editor.mjs` | **PASS**：12 项检查、页面错误 0；报告与截图在 `/tmp/dragon-s4-hot-fresh-20260925/` |

真实 HTTP 探针使用 `tests.manual_mask_hot_server` 在 `127.0.0.1:20543` 创建 `/tmp/anima-mask-hot-rx8unp_t`，Vite 在 `127.0.0.1:20544` 将 `/api` 代理至该服务。浏览器为 Headless Chrome 153；桌面 1440 x 1000、手机 390 x 844。探针验证画笔/橡皮擦黑白像素、撤销/重做/反选/缩放/平移、保存 PNG 并重载、批量应用 TOML、两标签陈旧 revision 的 409 与草稿保留、移动视口无整页溢出、预览/打标子集身份与返回路径。重载后画布非空像素为黑 9,155、白 450,861；未修改仓库静态发布包或用户数据。首次探针暴露第二标签加载等待不足；第二次复用已写入 fixture 导致初态断言不适用；调整等待并重建临时根后完整通过，前两次不计入通过率。
