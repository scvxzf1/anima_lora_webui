# Classic / Dragon / 新前端功能对照审计（第一轮）

状态：截图范围已审计 / 全站持续补充<br>
日期：2026-08-15

## 1. 审计口径

本轮按用户提供的四组界面逐项核对：

1. 项目预设、训练输出配置与 TOML 文件分组；
2. 必填、常用、预览、优化、高级配置目录；
3. 数据集蓝图、预设分组和 subset 工作区；
4. 训练历史统计、筛选、归档和分组工作台。

每项区分五个层面：Classic 行为基线、现有 vanilla Dragon、React/TypeScript
新前端、aiohttp 后端契约、自动化证据。后端 API 复用是目标架构的一部分；Classic
DOM、CSS、`anima-app` state、bridge 和 chunks 不得成为新前端运行时依赖。

## 2. 总结

- Classic 仍是最完整的功能清单，尤其是通用 TOML manager 和历史集合工作台。
- 现有 Dragon 已独立实现数据集主体、队列、监控和预览等大量页面，但并非完全独立。
- 旧 vanilla Dragon 的数据集分阶段调度仍动态导入 Classic state/bridge 和
  `stage-resolution-ui.js`；React/TypeScript 新前端已经有独立领域模型和 dialog，不能再把旧依赖误判为新前端缺口。
- 新前端第一个纵向功能切片已扩展到数据集编辑、预设库事务和图片预览：分组 CRUD、搜索、
  排序/跨组移动、导入/导出、完整 defaults/subset 表单、预设事务、图片/caption 预览和大图查看均已有类型化 API、Query/Mutation 和组件测试。
- 新前端已经实现独立分阶段调度、SPA route blocker 和数据集应用到共享训练上下文；通用 TOML manager、训练配置写入/启动、
  运行时、队列与历史工作台仍未迁移，因此 Classic 暂时不能退役。

## 3. 项目预设与 TOML 管理

| 能力 | Classic | 现有 Dragon | 新前端 | 后端契约 / 结论 |
| --- | --- | --- | --- | --- |
| 项目预设 / 输出配置双模式 | `web/static/index.html:106-109` | 配置页以 schema 表单为主 | 未实现 | `/api/config/output-runs*` 已存在 |
| 当前文件与训练/只读/dirty 状态 | `index.html:110-121`；`21-update-toml-selection-ui.js` | 有当前配置与 dirty 保存栏 | 未实现 | `/api/config/raw` 返回 meta |
| 保存、另存、直接编辑 | `index.html:123-141`；`22-update-toml-action-state.js` | 表单保存已实现，原始 TOML manager 不等价 | 未实现 | `PUT /api/config/raw`、`POST /raw/save-as` |
| 导入、导出、重新读取、删除、恢复 | Classic 全套存在 | 未见等价通用 manager | 未实现 | raw、file-group、restore-system API 已存在 |
| 文件锁与分组锁 | Classic 有文件/组锁 UI | 未见等价完整工作台 | 未实现 | `/api/config/lock`、`/group-lock` |
| 分组 CRUD、排序、跨组移动 | `toml-manager/drag-*` | 数据集页有部分 API 接线 | 数据集“创建组”迁移中 | `/file-groups`、`/place`、`/move-file` |

关键 Classic 证据：

- `web/static/js/features/toml-manager/mode.js:46-166`
- `web/static/js/features/toml-manager/actions.js:37-287`
- `web/static/js/features/toml-manager/drag-core.js:17-111`
- `web/static/js/features/toml-manager/drag-actions.js:21-231`
- `web/routes/config.py:84-100`

结论：截图中的项目预设工作台不能用当前 Dragon 配置表单替代，必须在新前端作为
独立 `training-config` feature 重建。

## 4. 配置目录与字段分类

| 能力 | Classic | 现有 Dragon | 新前端 | 结论 |
| --- | --- | --- | --- | --- |
| 必填 / 常用 / 预览 / 优化 / 高级 | `index.html:250-271` | `dragon-ui/category-map.js` 使用新的导航分类 | 未实现 | 需从共享 catalog 生成 typed schema |
| 字段 label、option、help | `config/catalog/*` | Dragon 复用 catalog/映射 | 未实现 | 业务文案可迁移，旧 renderer 不迁移 |
| 动态方法/模型族过滤 | Classic 配置表单已有 | Dragon 部分已有 | 未实现 | 必须补行为级契约测试 |
| 来源值、默认值、草稿与保存 | Classic 成熟 | Dragon 核心字段已有 | 未实现 | 不能只验证页面出现字段 |

权威字段来源：

- `web/static/js/config/catalog/form-layout.js:354-500`
- `web/static/js/config/catalog/labels-options.js:1-324`
- `web/static/js/config/catalog/field-help-dataset.js:3-149`
- `tests/test_frontend_config_search.py:8-38`

## 5. 数据集蓝图

| 能力 | Classic | 现有 Dragon | 新前端 | 后端 / 测试结论 |
| --- | --- | --- | --- | --- |
| 分组预设库 | 完整 | 已有独立 UI | 已实现首版 | `GET /api/config/dataset-presets` |
| 搜索与刷新 | 有 | 有 | 已实现并有组件测试 | 前端本地搜索 + Query refetch |
| 选择并读取预设详情 | 有 | 有 | 已实现并有组件测试 | `GET /dataset-presets/read` |
| 新建分组 | 有 | 有 API 调用 | 已实现并有 mutation 测试 | `POST /file-groups {kind: dataset}` |
| 重命名 / 删除分组 | 有 | 有 | 已实现，迁移中 | PATCH/DELETE 契约、不可操作组按钮和“删组保留 TOML”确认已覆盖 |
| 新建、保存、另存、复制、重命名、删除预设 | 有 | 有 | 已实现，迁移中 | overwrite、只读、dirty、重命名顺序和危险确认已有组件/API 契约测试；隔离数据浏览器写入验收待补 |
| 导入 / 导出 | 有 | 有 | 已实现，迁移中 | JSON 导入不覆盖、原始 TOML Blob 导出和 basename 已有组件/API 测试 |
| 组内排序 / 跨组拖动 | 有 | 有入口，等价性未证明 | 已实现，迁移中 | dnd-kit Pointer/Keyboard、箭头排序、空组 dropzone、跨组选择器共用 `/file-groups/place` 完整 order 契约 |
| 分组排序 | 有 | 未见等价 UI | 已实现，迁移中 | 仅 dataset scope 中可移动组参与；搜索、锁定、fixed 和 `unfiled_datasets` 不可排序 |
| subset 编辑与排序 | 有 | 有 | 已实现，迁移中 | RHF field array、dnd-kit Pointer/Keyboard、箭头/Alt+方向键和完整高级字段已接入 |
| subset 实验规则 scope | 有 | 未覆盖等价交互 | 已实现，迁移中 | NL/Tag 与 trigger clone 可显式批量应用到所选 subset，每行独立持久化 |
| 图片/caption 预览 | 有 | 有独立模块 | 已验证 | 独立 React 实现 saved/dirty guard、limit 120、请求取消/刷新、lazy image、失败态、caption 展示/复制；真实目录 60 图只读验收通过 |
| 通用大图查看 | 有 | Dragon 图片卡片实际不可点击 | 已验证 | 独立 React 全屏查看器支持图片/文件/caption 详情、失败重试、Esc 分层关闭、Tab 焦点陷阱和焦点恢复；桌面与 390×844 移动端边界验收通过 |
| 分阶段调度 | 有 | 有但旧依赖 | 已验证 | React 独立 `StageScheduleEditor` + 纯领域模型已覆盖模板、连续区间、subset 绑定、校验与保存 payload；零 bridge import |
| dirty / 只读 / 危险操作 | 有 | 有 | 已验证 | 切换 guard、`beforeunload`、React Router blocker、只读按钮和删除确认均已接入；测试覆盖导航取消/确认 |

现有 Dragon 的旧依赖证据：

- `web/static/js/dragon-ui/pages/dataset-editor.js:550-587` 动态导入
  `anima-app` bridge/state 和 `stage-resolution-ui.js`。
- `dataset-editor.js:655-688` 把 Dragon 草稿同步回 legacy state。

新前端首个切片证据：

- `web/frontend-next/src/features/dataset-editor/api.ts`
- `web/frontend-next/src/features/dataset-editor/DatasetWorkspace.tsx`
- `web/frontend-next/src/features/dataset-editor/DatasetPreviewDialog.tsx`
- `web/frontend-next/src/features/dataset-editor/DatasetImageViewer.tsx`
- `web/frontend-next/src/features/dataset-editor/StageScheduleEditor.tsx`
- `web/frontend-next/src/features/dataset-editor/useUnsavedChangesGuard.ts`
- `web/frontend-next/src/features/dataset-editor/api.test.ts`
- `web/frontend-next/src/features/dataset-editor/DatasetWorkspace.test.tsx`

## 6. 训练历史工作台

| 能力 | Classic | 现有 Dragon | 新前端 | 结论 |
| --- | --- | --- | --- | --- |
| 总量、训练、预处理、异常、归档、队列统计 | 有 | 基础历史页未等价覆盖 | 未实现 | 需 typed history summary model |
| 全局搜索与多维筛选 | 有 | 搜索 + 状态筛选 | 未实现 | Classic 筛选面更完整 |
| 集合 / 配置组分组 | 有 | 未见等价工作台 | 未实现 | 后端 collections API 已存在 |
| 批量归档、取消归档、删除、设置集合 | 有 | 未见等价覆盖 | 未实现 | `/api/training/history/batch` |
| 详情、日志、样张、权重、续训 | 有 | 基础能力已有 | 未实现 | 可作为第二阶段行为参考 |
| 时间线与拖动归类 | 有 | 未见等价覆盖 | 未实现 | 需继续审计 API 和排序语义 |

关键证据：

- `web/static/index.html:443-490`
- `web/static/js/features/history-list/task-collections.js:66-381`
- `web/static/js/features/history-list/collections-workbench.js`
- `web/static/js/dragon-ui/pages/history.js:16-63`
- `web/routes/training.py:42-52,637-673`

## 7. 新前端基础设施审计

- 新应用位于 `web/frontend-next/`，使用 React、TypeScript、Vite、TanStack Query、
  React Hook Form、Zod 和 React Router。
- `web/server.py` 同时服务 `/next` 与 `/next/{path:.*}`，允许 BrowserRouter 深链刷新。
- 前端导航使用 Router link，不触发整页请求。
- 当前新功能树未 import `web/static/js/features/anima-app`、Classic DOM 或 bridge。
- `web/frontend-next/src` 对 `anima-app`、`legacyStage`、`stage-resolution-ui`、bridge 和 Classic 引用扫描为零命中。
- 当前前端门禁为 46 个测试通过，覆盖预览 URL、saved/dirty/readonly guard、SPA blocker、
  browser unload、阶段调度保存 payload、数据集应用目标/dirty guard、基础 dialog Escape/焦点恢复、caption 复制、lazy image、失败态、刷新和全屏。
- 图片预览相关后端定向门禁为 59 个测试通过，先前数据集/配置较宽门禁为 114 个测试通过；真实浏览器已验收深链加载、完整编辑表单和真实 60 图预览。
- 浏览器验收覆盖桌面与 390×844 移动端：大图 Esc 后焦点恢复到图片按钮，预览 Esc 后恢复到 subset 预览按钮；移动端 `bodyScrollWidth == bodyWidth`，图片完全包含于画布且详情无覆盖。全程未保存、导入、删除或改写用户数据。

## 8. 下一轮唯一优先事项

阶段 3 已将 `/next/training` 推进到来源感知编辑、后端变更预览、差异保存、另存防覆盖、dirty guard 与结构化 preflight。下一步补方法/模型族动态过滤、完整字段面、配置冻结和危险确认，再开放安全启动训练；历史页面继续排在训练配置/启动之后。
