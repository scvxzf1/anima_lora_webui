# anima_lora WebUI 启动指南

本文只说明如何在 **Linux** 上把 WebUI 跑起来。

<img width="2483" height="1448" alt="image" src="https://github.com/user-attachments/assets/a1e63f86-f6ac-4baf-a8dd-ceea65334b7d" />
<img width="2483" height="1448" alt="image" src="https://github.com/user-attachments/assets/9d534da9-5df6-4915-b453-8bb40f428396" />

如果你只是想先看到页面，按下面步骤执行即可。


## 1. 环境要求

推荐环境：

- Linux x86_64
- NVIDIA 显卡与可用驱动
- `nvidia-smi` 可以正常显示显卡
- 网络可以访问 GitHub、PyPI、PyTorch wheel 源和 Hugging Face

先检查显卡：

```bash
nvidia-smi
```


## 2. 安装系统依赖

Ubuntu / Debian 系统执行：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl wget build-essential \
  python3 python3-venv python3-pip \
  libgl1 libglib2.0-0
```

启用 Git LFS：

```bash
git lfs install
```


## 3. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

如果提示找不到 `uv`，重新打开终端后再试。


## 4. 下载项目

```bash
git clone <你的仓库地址> anima_lora
cd anima_lora
```

如果仓库使用了 Git LFS：

```bash
git lfs pull
```


## 5. 安装 Python 环境

在项目根目录执行：

```bash
uv sync
```

完成后会生成：

```text
.venv/
```

验证环境：

```bash
.venv/bin/python --version
.venv/bin/python -c "import toml, aiohttp, accelerate; print('ok')"
```


## 6. 启动 WebUI

推荐启动：

```bash
.venv/bin/python -m web --host 127.0.0.1 --port 20103
```

浏览器打开：

```text
http://127.0.0.1:20103/
```

看到页面后，WebUI 就已经跑起来了。


## 7. 局域网访问

如果需要让同一局域网内的其他设备访问：

```bash
.venv/bin/python -m web --host 0.0.0.0 --port 20103
```

然后在其他设备浏览器打开：

```text
http://服务器IP:20103/
```

注意：`0.0.0.0` 会把 WebUI 暴露到当前网络，请只在可信网络中使用。


## 8. 后台运行

```bash
mkdir -p logs
nohup .venv/bin/python -m web --host 127.0.0.1 --port 20103 \
  > logs/webui-20103.log 2>&1 &
```

查看日志：

```bash
tail -f logs/webui-20103.log
```

查看端口：

```bash
ss -ltnp | grep 20103
```

停止 WebUI：

```bash
pkill -f "python -m web"
```


## 9. 常见问题

### ModuleNotFoundError: No module named 'toml'

说明你没有使用项目虚拟环境。

请使用：

```bash
.venv/bin/python -m web --host 127.0.0.1 --port 20103
```

不要直接使用：

```bash
python -m web
```


### 端口已被占用

检查端口：

```bash
ss -ltnp | grep 20103
```

换一个端口：

```bash
.venv/bin/python -m web --host 127.0.0.1 --port 20104
```


### uv sync 下载很慢或失败

通常是网络问题。确认当前机器可以访问：

- GitHub
- PyPI
- PyTorch wheel 源

网络恢复后重新执行：

```bash
uv sync
```


## 10. 后续说明

更完整的 Linux 部署说明在：

```text
docs/guidelines/linux-deployment.zh.md
```

