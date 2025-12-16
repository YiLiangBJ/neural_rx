# Neural RX - 快速开始

**⚠️ 系统要求: Linux + Python 3.10**

根据你的硬件配置,选择对应的安装方式:

## 🖥️ 场景选择

### 1️⃣ Linux CPU (开发和小规模实验)
```bash
source .env  # 如需代理
uv sync --extra cpu
source .venv/bin/activate
```

### 2️⃣ Linux GPU (生产训练 - 推荐)
```bash
source .env  # 如需代理
uv sync --extra gpu
source .venv/bin/activate
```

## ⚙️ 环境要求

- **操作系统**: Linux (推荐 Ubuntu 22.04)
- **Python**: 3.10 (`.python-version` 已配置,UV 会自动下载)
- **GPU**: 推荐 NVIDIA GPU + CUDA 12.x (用于 `--extra gpu`)

**注意**: 
- ❌ **不支持 Windows** (Sionna/Mitsuba 不兼容)
- ❌ **不支持 macOS** (TensorRT 不支持)

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
