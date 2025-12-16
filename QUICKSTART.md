# Neural RX - 快速开始

根据你的机器环境,选择对应的安装方式:

## 🖥️ 场景选择

### 1️⃣ Windows CPU (代码调试)
```bash
source .env
uv sync --extra windows-cpu
source .venv/Scripts/activate
```

### 2️⃣ Linux CPU (训练评估)
```bash
source .env
uv sync --extra linux-cpu
source .venv/bin/activate
```

### 3️⃣ Linux GPU (完整功能)
```bash
source .env
uv sync --extra linux-gpu
source .venv/bin/activate
```

## ⚙️ 环境要求

- **Python**: 3.10 (官方推荐,`.python-version` 已配置)
- **操作系统**: 
  - Windows: 仅支持 CPU 调试 (TF 2.10.1 + Sionna 0.14)
  - Linux: 支持 CPU 和 GPU (TF 2.15 + Sionna 0.18,官方推荐)

**注意**: Windows 和 Linux 的 TensorFlow/Sionna 版本不同是因为 TF 2.15+ 不支持 Windows。

## 📖 详细文档

查看 `SETUP.md` 获取完整安装指南和故障排除。

## 🚀 快速验证

```bash
# 运行完整系统验证脚本(推荐)
python verify_gpu.py

# 输出包括:
# - 系统信息(OS、Python 版本)
# - CPU 信息(核心数、频率、使用率)
# - 内存信息(总量、可用、SWAP)
# - 磁盘信息(各分区容量)
# - GPU 检测(TensorFlow/PyTorch)
# - CUDA/cuDNN 版本
# - 性能评估和使用建议

# 或者手动检查
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
python -c "import sionna as sn; print(f'Sionna: {sn.__version__}')"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## 🔗 相关链接

- [完整设置文档](SETUP.md)
- [Sionna 官方文档](https://nvlabs.github.io/sionna/)
- [项目 README](README.md)
