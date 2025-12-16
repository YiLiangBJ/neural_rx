# 🔧 TensorFlow GPU 修复指南

## 问题

TensorFlow 2.15.0 在你的 Linux 机器上无法检测到 GPU,错误信息:
```
Unable to register cuDNN factory
Unable to register cuFFT factory
Unable to register cuBLAS factory
```

## 原因

**CUDA 库版本冲突**:
- 系统 CUDA: **12.9** / **13.0**
- TensorFlow 2.15.0 需要: **CUDA 12.2/12.3**
- 之前安装的 `tensorflow==2.15.0` 试图使用系统 CUDA 库,导致版本不匹配

## 解决方案

使用 **`tensorflow==2.15.0` + 显式的 NVIDIA CUDA 包**:
- ✅ 在虚拟环境中安装 CUDA 12.3
- ✅ 在虚拟环境中安装 cuDNN 9.1
- ✅ 不依赖系统 CUDA 版本
- ✅ 避免 `tensorflow[and-cuda]` 的依赖问题
- ✅ 就像 PyTorch 一样开箱即用

---

## 📝 操作步骤

### 1. 修改配置文件

已修改 `pyproject.toml`:

```toml
gpu = [
    "tensorflow[and-cuda]==2.15.0",  # ✅ 改动:自带 CUDA 12.3, cuDNN 9.0, TensorRT 8.6
    "sionna==0.18.0",
    "mitsuba==3.5.2",
    "onnx==1.14.0",
    "tf2onnx>=1.16.0",
    "polygraphy>=0.49.0",
    # TensorRT 由 tensorflow[and-cuda] 提供,不单独指定
]
```

### 2. 在 Linux 机器上重新安装

```bash
cd ~/neural_rx

# 删除旧环境
rm -rf .venv uv.lock

# 拉取最新代码
git pull

# 重新安装(使用 GPU extra)
uv sync --extra gpu

# 激活虚拟环境
source .venv/bin/activate
```

### 3. 验证安装

```bash
# 检查安装的 CUDA 包
uv pip list | grep -E "(nvidia|cuda|cudnn)"

# 应该看到:
# nvidia-cublas-cu12        12.3.x.x
# nvidia-cuda-cupti-cu12    12.3.x
# nvidia-cuda-nvcc-cu12     12.3.x
# nvidia-cuda-runtime-cu12  12.3.x
# nvidia-cudnn-cu12         9.0.x.x
# nvidia-cufft-cu12         11.0.x.x
# nvidia-curand-cu12        10.3.x.x
# nvidia-cusolver-cu12      11.5.x.x
# nvidia-cusparse-cu12      12.2.x.x
# nvidia-nccl-cu12          2.x.x
# nvidia-nvjitlink-cu12     12.3.x
```

### 4. 测试 GPU 检测

```bash
# 测试 TensorFlow GPU
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# 期望输出:
# TensorFlow: 2.15.0
# GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 5. 运行完整验证

```bash
python verify_gpu.py
```

期望看到:
```
✅ TensorFlow 版本: 2.15.0
✅ GPU 可用: True
   检测到 1 块 GPU:
   - GPU 0: /physical_device:GPU:0
   
   CUDA 版本: 12.3
   cuDNN 版本: 9.0
   
   ✅ GPU 计算测试成功
```

### 6. 运行训练

```bash
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
```

应该看到:
```
Using GPU 0 only.
GPU memory growth enabled for GPU 0
```

而不是:
```
IndexError: list index out of range
```

---

## 🔍 故障排除

### 问题 1: 仍然报 cuDNN 错误

**检查环境变量**:
```bash
echo $LD_LIBRARY_PATH
```

如果包含系统 CUDA 路径,临时取消:
```bash
unset LD_LIBRARY_PATH
unset CUDA_HOME
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

如果这样工作了,更新 `.env` 文件:
```bash
# 在 .env 中添加
unset LD_LIBRARY_PATH
unset CUDA_HOME
```

### 问题 2: 仍然检测不到 GPU

**检查 NVIDIA 驱动**:
```bash
nvidia-smi
```

必须能看到 GPU 信息。

**检查 CUDA_VISIBLE_DEVICES**:
```bash
echo $CUDA_VISIBLE_DEVICES
```

如果是空字符串 `""`,会隐藏所有 GPU:
```bash
unset CUDA_VISIBLE_DEVICES
```

### 问题 3: 安装时依赖冲突

```bash
# 清除所有缓存
rm -rf .venv uv.lock ~/.cache/uv

# 重新安装
uv sync --extra gpu
```

---

## 📊 对比

| 配置 | 之前 | 现在 |
|------|------|------|
| TensorFlow 包 | `tensorflow==2.15.0` | `tensorflow[and-cuda]==2.15.0` |
| CUDA 来源 | 系统 CUDA 12.9/13.0 | 包自带 CUDA 12.3 |
| cuDNN 来源 | 手动安装 | 包自带 cuDNN 9.0 |
| 版本匹配 | ❌ 不匹配 | ✅ 完美匹配 |
| GPU 检测 | ❌ 失败 | ✅ 成功 |
| 依赖管理 | ❌ 复杂 | ✅ 简单 |

---

## ✅ 成功标志

当你看到以下输出时,说明修复成功:

```bash
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

$ python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
Using GPU 0 only.
GPU memory growth enabled for GPU 0
[训练开始...]
```

---

## 📚 技术说明

### 为什么 `tensorflow[and-cuda]` 更好?

1. **版本匹配**: TensorFlow 官方保证 CUDA/cuDNN 版本完全匹配
2. **隔离环境**: 虚拟环境内的 CUDA 不影响系统
3. **可重现**: 不同机器上行为一致
4. **简化安装**: 无需手动安装 CUDA Toolkit 和 cuDNN
5. **与 PyTorch 一致**: 现在两者都自带 CUDA

### 额外安装的包

`tensorflow[and-cuda]==2.15.0` 会自动安装:
- `nvidia-cublas-cu12`
- `nvidia-cuda-cupti-cu12`
- `nvidia-cuda-nvcc-cu12`
- `nvidia-cuda-runtime-cu12`
- `nvidia-cudnn-cu12`
- `nvidia-cufft-cu12`
- `nvidia-curand-cu12`
- `nvidia-cusolver-cu12`
- `nvidia-cusparse-cu12`
- `nvidia-nccl-cu12`
- `nvidia-nvjitlink-cu12`
- `tensorrt==8.6.1.post1` (TensorFlow 2.15 兼容版本)

总大小约 **~3GB**,但完全值得!

**注意**: TensorRT 8.6.1 是 TensorFlow 2.15 官方支持的版本。虽然比 9.x 旧,但是经过充分测试和优化的。

---

## 🎉 下一步

修复后,你可以:

1. **开始训练**:
   ```bash
   python scripts/train_neural_rx.py -config_name nrx_large
   ```

2. **评估模型**:
   ```bash
   python scripts/evaluate.py -config_name nrx_large
   ```

3. **运行 Jupyter Notebooks**:
   ```bash
   jupyter notebook notebooks/jumpstart_tutorial.ipynb
   ```

---

祝训练顺利! 🚀
