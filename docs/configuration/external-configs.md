# 配置目录外置功能

## 概述

anima_lora 现在支持将配置目录外置到独立位置，便于：
- 多实例部署时隔离配置
- 容器化环境中分离系统配置和用户数据
- 将运行时数据（训练历史、队列）从代码仓库分离
- 在不同磁盘上管理配置和数据

## 配置方式

### 方法 1：环境变量（推荐）

创建 `.env` 文件（项目根目录）：

```bash
# 配置根目录（包含 base.toml, methods/, datasets/, web-training-history/, web-training-queue/ 等）
ANIMA_CONFIGS_ROOT=/path/to/configs
```

### 方法 2：WebUI 全局设置

在 WebUI 的"全局设置"页面配置：
1. 打开"路径与默认模型"部分
2. 在"配置目录路径"卡片中设置"配置根目录"
3. 保存设置

## 路径解析规则

- **相对路径**：相对于项目根目录（anima_lora/）解析
- **绝对路径**：直接使用
- **环境变量扩展**：支持 `$HOME`、`~` 等扩展
- **安全检查**：自动拒绝包含 `..` 的路径（防止路径遍历）

## 使用示例

### 示例 1：外置到用户主目录

```bash
# .env
ANIMA_CONFIGS_ROOT=$HOME/.local/share/anima/configs
```

### 示例 2：外置到数据盘

```bash
# .env
ANIMA_CONFIGS_ROOT=/mnt/data/anima/configs
```bash
# .env
ANIMA_CONFIGS_ROOT=/mnt/data/anima/configs
```

### 示例 3：容器化部署

```bash
# .env
ANIMA_CONFIGS_ROOT=/etc/anima/configs
```

**Docker Compose 示例**：

```yaml
services:
  anima-lora:
    image: anima-lora:latest
    environment:
      ANIMA_CONFIGS_ROOT: /etc/anima/configs
    volumes:
      - ./configs:/etc/anima/configs:ro

volumes:
  configs:
```

## 默认行为

未设置任何环境变量时，保持原有行为：
- 配置根目录：`anima_lora/configs`
- 包含所有配置文件、训练历史和队列

无需任何迁移，完全向后兼容。

## 环境变量优先级

配置解析优先级（从高到低）：
1. `ANIMA_CONFIGS_ROOT` 环境变量
2. WebUI 全局设置中的 `configs_root`
3. 默认 `configs/` 目录

## 迁移指南

如果已有 `configs/` 目录数据，可以：

### 选项 1：使用符号链接（简单）

```bash
# 1. 移动 configs 到新位置
mv configs /path/to/new/configs

# 2. 创建符号链接
ln -s /path/to/new/configs configs

# 3. 设置环境变量（可选）
echo "ANIMA_CONFIGS_ROOT=/path/to/new/configs" >> .env
```

### 选项 2：完整外置（推荐）

```bash
# 1. 复制整个 configs 目录
cp -r configs /path/to/new/configs

# 2. 设置环境变量
echo "ANIMA_CONFIGS_ROOT=/path/to/new/configs" > .env

# 3. 验证并备份原目录
python test_config_migration.py
mv configs configs.backup
```

## 故障排查

### 检查当前配置

运行测试脚本查看实际使用的路径：

```bash
python test_config_migration.py
```

### 常见问题

**Q: 设置环境变量后路径没有变化？**

A: 确保 `.env` 文件在项目根目录，并重启 WebUI 或训练进程。

**Q: 相对路径相对于哪里？**

A: 所有相对路径都相对于项目根目录（`anima_lora/`），而不是当前工作目录。

**Q: 外置后原 configs/ 目录会被删除吗？**

A: 不会。外置是通过环境变量切换路径，不会自动删除任何文件。

## 性能影响

路径解析在模块初始化时执行一次，运行时没有额外性能开销。

## 安全性

- 自动加载 `.env` 文件时不会覆盖已有的环境变量
- `.env` 文件已被 `.gitignore` 排除，不会意外提交敏感路径
- 路径安全检查防止 `..` 遍历攻击
- 建议将外置配置目录设置适当的文件权限

## 技术细节

### 实现位置

- `library/env.py` - 路径获取和环境变量解析
- `web/services/settings_service.py` - 全局设置管理
- `web/services/training_service.py` - 训练服务路径
- `.env.example` - 环境变量模板

### 测试覆盖

- `tests/test_env_config_paths.py` - 路径解析单元测试
- `test_config_migration.py` - 端到端验证脚本

## 反馈和支持

如有问题或建议，请提交 Issue 或 Pull Request。
