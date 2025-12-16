# Neural RX 环境设置指南

本项目使用 `uv` 管理 Python 依赖,支持三种不同的使用场景。

## 前置要求

- **uv** 包管理器 (安装: `pip install uv` 或参考 https://github.com/astral-sh/uv)
- **Python 3.10** (官方推荐版本)
- **Git** (用于克隆代码)

---

## 📦 三种使用场景

### 场景 1: Windows CPU (简单调试)

**适用于**: 在 Windows 机器上进行代码开发和简单调试,无 GPU

**限制**:
- 使用 TensorFlow 2.10.1 (Windows 最后支持版本)
- 使用 Sionna 0.14.0 (兼容版本)
- 不支持 Mitsuba 射线追踪和 TensorRT 加速

**安装步骤**:

```bash
# 1. 加载代理配置 (如需要)
source .env

# 2. 删除旧虚拟环境 (如果存在)
rm -rf .venv

# 3. 创建虚拟环境并安装依赖
uv sync --extra windows-cpu

# 4. 激活虚拟环境 (Git Bash)
source .venv/Scripts/activate

# 或者 (PowerShell)
.\.venv\Scripts\Activate.ps1
```

**验证安装**:
```python
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sionna as sn; print(f'Sionna: {sn.__version__}')"
```

---

### 场景 2: Linux CPU (训练和评估,无 GPU)

**适用于**: 在 Linux 服务器上进行训练和评估,但没有 GPU

**包含**:
- TensorFlow 2.15.0 CPU 版本 (官方推荐)
- Sionna 0.18.0 (官方推荐)
- Mitsuba 3.5.2 (射线追踪信道模拟)

**安装步骤**:

```bash
# 1. 加载代理配置 (如需要)
source .env

# 2. 删除旧虚拟环境 (如果存在)
rm -rf .venv

# 3. 创建虚拟环境并安装依赖
uv sync --extra linux-cpu

# 4. 激活虚拟环境
source .venv/bin/activate
```

**验证安装**:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sionna as sn; print(f'Sionna: {sn.__version__}')"
python -c "import mitsuba as mi; print(f'Mitsuba: {mi.__version__}')"
```

---

### 场景 3: Linux GPU (完整训练和评估,带 GPU)

**适用于**: 在带有 NVIDIA GPU 的 Linux 服务器上进行完整训练和评估

**包含**:
- TensorFlow 2.15.0 with CUDA 支持 (官方推荐)
- Sionna 0.18.0 (官方推荐)
- Mitsuba 3.5.2 (射线追踪信道模拟)
- TensorRT 10.x+ (推理加速)
- CUDA 12.x 支持

**前置要求**:
- NVIDIA GPU 驱动 (推荐 >= 525.x)
- 已安装 CUDA Toolkit (uv 会自动安装 Python CUDA 包)

**安装步骤**:

```bash
# 1. 加载代理配置 (如需要)
source .env

# 2. 删除旧虚拟环境 (如果存在)
rm -rf .venv

# 3. 创建虚拟环境并安装依赖
uv sync --extra linux-gpu

# 4. 激活虚拟环境
source .venv/bin/activate
```

**验证安装**:
```bash
# 检查 TensorFlow GPU
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# 检查 Sionna
python -c "import sionna as sn; print(f'Sionna: {sn.__version__}')"

# 检查 TensorRT
python -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"
```

---

## 🚀 快速开始

### 运行训练脚本

```bash
# 激活环境后
cd scripts
python train_neural_rx.py -config_name <config_file>
```

### 运行评估脚本

```bash
cd scripts
python evaluate.py -config_name <config_file> -gpu 0
```

### 启动 Jupyter Notebook

```bash
jupyter notebook
# 或
jupyter lab
```

---

## 🔧 代理配置 (Intel 内网用户)

如果你在 Intel 内网,需要配置代理。`.env` 文件已包含代理设置:

```bash
# 查看 .env 内容
cat .env

# 加载代理
source .env
```

---

## 📝 常见问题

### Q1: Windows 上遇到 `tensorflow-io-gcs-filesystem` 错误?

**原因**: TensorFlow 2.15+ 不支持 Windows。

**解决**: 使用 `windows-cpu` 场景,会自动安装 TensorFlow 2.10.1。

```bash
uv sync --extra windows-cpu
```

### Q2: 如何切换不同的环境?

```bash
# 删除当前虚拟环境
rm -rf .venv

# 安装新环境
uv sync --extra <windows-cpu|linux-cpu|linux-gpu>
```

### Q3: 如何更新依赖?

```bash
# 更新所有依赖到最新版本
uv sync --upgrade

# 更新特定包
uv pip install --upgrade <package-name>
```

### Q4: 如何添加新的依赖?

```bash
# 添加到核心依赖 (所有环境)
uv add <package-name>

# 添加到特定环境组
# 需要手动编辑 pyproject.toml 的 [project.optional-dependencies] 部分
```

### Q5: Linux 上缺少 CUDA 驱动?

安装 NVIDIA 驱动和 CUDA Toolkit:

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install nvidia-driver-535  # 或更新版本
sudo reboot

# 验证
nvidia-smi
```

---

## 📚 参考资料

- [Sionna 官方文档](https://nvlabs.github.io/sionna/)
- [TensorFlow 安装指南](https://www.tensorflow.org/install)
- [uv 文档](https://github.com/astral-sh/uv)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)

---

## 🆘 获取帮助

遇到问题?检查以下内容:

1. Python 版本是否为 3.10? (官方推荐)
   ```bash
   python --version
   ```

2. uv 是否正确安装?
   ```bash
   uv --version
   ```

3. 代理配置是否正确?
   ```bash
   echo $HTTP_PROXY
   echo $HTTPS_PROXY
   ```

4. GPU 是否正确识别? (Linux GPU 环境)
   ```bash
   nvidia-smi
   ```
