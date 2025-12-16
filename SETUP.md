# Neural RX 环境设置指南

**⚠️ 系统要求: Linux + Python 3.10**

本项目使用 `uv` 管理 Python 依赖,仅支持 Linux 平台。

## 前置要求

- **操作系统**: Linux (推荐 Ubuntu 22.04 LTS)
- **uv** 包管理器
  ```bash
  pip install uv
  # 或
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Python 3.10** - ✅ UV 会自动下载安装
- **Git** (用于克隆代码)
- **代理配置** (可选,如在防火墙后)
- **NVIDIA GPU** (可选,用于 GPU 训练)

---

## 📦 两种使用场景

### 🖥️ CPU 环境 (开发和小规模实验)

**适用于**:
- 代码开发和调试
- 小规模实验
- 无 GPU 的服务器

**包含**:
- TensorFlow 2.15 (CPU)
- Sionna 0.18
- Mitsuba 3.5.2
- ONNX 1.14

**安装步骤**:

```bash
# 1. 克隆代码(如果还没有)
git clone https://github.com/YiLiangBJ/neural_rx.git
cd neural_rx

# 2. 加载代理配置 (可选)
source .env

# 3. 创建虚拟环境并安装依赖
uv sync --extra cpu

# 4. 激活虚拟环境
source .venv/bin/activate

# 5. 验证安装
python verify_gpu.py
```

---

### 🚀 GPU 环境 (生产训练 - 推荐)

**适用于**:
- 大规模神经接收器训练
- 端到端系统训练
- 推理加速

**前置要求**:
- NVIDIA GPU (推荐 RTX 3090, A100, H100)
- CUDA 12.x
- cuDNN 9.0+

**包含**:
- TensorFlow 2.15 (CUDA 支持)
- Sionna 0.18
- Mitsuba 3.5.2
- ONNX 1.14
- TensorRT 9.6+
- NVIDIA cuDNN 12

**安装步骤**:

```bash
# 1. 克隆代码(如果还没有)
git clone https://github.com/YiLiangBJ/neural_rx.git
cd neural_rx

# 2. 加载代理配置 (可选)
source .env

# 3. 创建虚拟环境并安装依赖(包含 TensorRT)
uv sync --extra gpu

# 4. 激活虚拟环境
source .venv/bin/activate

# 5. 验证安装
python verify_gpu.py
```

**预期输出**:
```
============================================================
检查 TensorFlow GPU 支持
============================================================
✅ TensorFlow 版本: 2.15.0
✅ GPU 可用: True
   检测到 1 块 GPU:
   - GPU 0: /physical_device:GPU:0
   CUDA 版本: 12.3
   cuDNN 版本: 9.0
   ✅ GPU 计算测试成功
```

---

## 🔄 切换环境

如果需要在 CPU 和 GPU 环境之间切换:

```bash
# 删除当前环境
rm -rf .venv uv.lock

# 安装新环境
uv sync --extra cpu   # 或 --extra gpu

# 激活
source .venv/bin/activate
```

---

## 📝 配置代理 (可选)

如果在防火墙后(如公司内网),编辑 `.env`:

```bash
# 复制示例文件
cp .env.example .env

# 编辑配置
nano .env

# 添加代理
export HTTP_PROXY=http://proxy-server:port
export HTTPS_PROXY=http://proxy-server:port
```

然后:
```bash
source .env
uv sync --extra gpu
```

---

## 🧪 快速测试

### 验证 Python 和基础包

```bash
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sionna as sn; print(f'Sionna: {sn.__version__}')"
```

### 验证 GPU (仅 GPU 环境)

```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 运行完整验证

```bash
python verify_gpu.py
```

---

## 🚀 开始使用

### 训练神经接收器

```bash
# 使用配置文件训练
python scripts/train_neural_rx.py -config_name nrx_large

# 查看所有可用配置
ls config/*.cfg
```

### 评估模型

```bash
# 评估训练好的模型
python scripts/evaluate.py -config_name nrx_large

# 查看结果
ls results/nrx_large_results/
```

### 运行 Jupyter Notebooks

```bash
jupyter notebook notebooks/jumpstart_tutorial.ipynb
```

---

## ❓ 故障排除

### 问题 1: Python 版本不对

```bash
python --version  # 应该显示 Python 3.10.x
```

**解决**: UV 会自动管理 Python 版本,确保 `.python-version` 文件存在。

### 问题 2: UV 下载失败

```bash
# 检查代理配置
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 重新加载
source .env
```

### 问题 3: GPU 未检测到

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA
nvcc --version

# 检查 TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 问题 4: 包冲突

```bash
# 删除所有缓存和重新开始
rm -rf .venv uv.lock
uv cache clean
uv sync --extra gpu
```

### 问题 5: Mitsuba 导入失败

Mitsuba 仅在 Linux 上支持,确保使用 Linux 系统。

---

## 📚 相关文档

- [快速开始](QUICKSTART.md)
- [配置总结](CONFIGURATION_SUMMARY.md)
- [GPU 验证脚本说明](docs/verify_gpu_usage.md)
- [UV Python 管理](docs/uv_python_management.md)

---

## 🔗 外部资源

- [Sionna 官方文档](https://nvlabs.github.io/sionna/)
- [TensorFlow GPU 支持](https://www.tensorflow.org/install/gpu)
- [UV 包管理器](https://github.com/astral-sh/uv)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

---

## ⚠️ Windows / macOS 用户

**此项目不支持 Windows 或 macOS**,原因:

- ❌ Sionna 依赖 Mitsuba,不支持 Windows/macOS
- ❌ TensorRT 仅支持 Linux
- ❌ TensorFlow 2.15+ GPU 仅支持 Linux
- ❌ 性能和兼容性问题

**建议**:
- 使用 Linux 服务器或工作站
- 使用 WSL2 (Windows Subsystem for Linux)
- 使用 Docker 容器
- 使用云端 GPU (AWS, Google Cloud, Azure)

---

祝使用顺利! 🎉

如有问题,请查看 [故障排除](#-故障排除) 或提交 Issue。
