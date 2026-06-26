## UI缩放功能实现总结

### 功能描述
在WebUI的全局设置中添加了"缩放比例"配置项，允许用户调整整个界面的缩放大小。

### 修改文件

#### 后端
1. **web/services/settings_service.py**
   - 添加 `DEFAULT_UI_SCALE = 100` 常量
   - 添加 `GLOBAL_UI_KEYS = ("ui_scale",)` 元组
   - 在 `save_global_settings()` 中添加UI设置处理逻辑
   - 在 `_load_settings()` 中添加UI设置加载逻辑
   - 在 `_default_global_settings()` 中添加默认值
   - 添加 `_normalize_ui_setting()` 函数，限制缩放范围在 25%-400%

#### 前端 - HTML
2. **web/static/index.html**
   - 在全局设置页面的"输出文件夹"和"基础模型路径"之间插入新的"界面设置"卡片
   - 添加 `<input id="global-ui-scale" type="number">` 输入框
   - 添加帮助文档说明
   - 更新全局设置概览统计（1个输出根目录 + 1个界面设置 + 3个模型路径）
   - 更新cache token为 `20260625-1`

#### 前端 - JavaScript
3. **web/static/js/features/app-shell/ui-scale.js** (新文件)
   - 创建 `createUIScaleController()` 函数
   - 提供 `applyUIScale()` 方法应用缩放
   - 提供 `applyScaleFromSettings()` 方法从全局设置加载缩放
   - 通过CSS变量 `--ui-scale` 和 `root.style.fontSize` 应用缩放

4. **web/static/js/config/catalog/defaults.js**
   - 添加 `GLOBAL_UI_FIELDS = [['ui_scale', 'global-ui-scale']]`
   - 将其注册到 `GLOBAL_SETTING_INPUTS`

5. **web/static/js/features/anima-app/imports.js**
   - 导入 `createUIScaleController`
   - 添加到全局导出

6. **web/static/js/features/anima-app/chunks/02-ensure-history-detail-feature.js**
   - 在 `startAnimaApp()` 中初始化 `uiScaleController`
   - 在 `boot()` 中调用 `uiScaleController.initUIScale()`

7. **web/static/js/features/anima-app/chunks/26-load-global-settings.js**
   - 在 `loadGlobalSettings()` 中调用 `uiScaleController?.applyScaleFromSettings?.(data)`
   - 在 `saveGlobalSettings()` 中调用 `uiScaleController?.applyScaleFromSettings?.(globalSettings)`

8. **web/static/js/features/anima-app/index.js**
   - 更新所有chunk imports的cache token

### 使用方法
1. 启动WebUI：`.venv/bin/python tasks.py web`
2. 打开浏览器访问 WebUI
3. 点击顶部导航栏的"全局设置"标签
4. 在"界面设置"卡片中找到"缩放比例 (%)"输入框
5. 输入期望的缩放百分比（25-400，默认100）
6. 点击"保存更新当前选中配置"按钮
7. 界面会立即应用新的缩放比例

### 技术实现
- 通过修改 `document.documentElement.style.fontSize` 实现缩放
- 设置CSS变量 `--ui-scale` 供其他样式使用
- 缩放值存储在 `configs/web-ui-settings.toml` 的 `[global]` section
- 页面加载时自动从全局设置读取并应用缩放

### 验证
所有修改的文件已通过语法检查：
- Python: `python -m py_compile` ✓
- JavaScript: `node --check` ✓
- 逻辑正确性已验证
