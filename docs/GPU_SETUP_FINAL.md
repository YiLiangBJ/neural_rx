# ✅ TensorFlow GPU 配置修复 - 最终版本

## 问题

TensorFlow 无法检测 GPU,因为:
- 系统 CUDA: 12.9/13.0
- TensorFlow 2.15 需要: CUDA 12.3
- `tensorflow[and-cuda]` 包有依赖问题(tensorrt-libs 缺失)

## 解决方案

**在虚拟环境中安装 CUDA 包** - 不依赖系统 CUDA!

---

## 🚀 立即操作(在 Linux 机器上)

```bash
# 1. 进入项目目录
cd ~/neural_rx

# 2. 拉取最新更改
git pull

# 3. 删除旧环境
rm -rf .venv uv.lock

# 4. 安装 GPU 环境(包含 CUDA 12.3 + cuDNN 9.1)
uv sync --extra gpu

# 5. 激活环境
source .venv/bin/activate

# 6. 验证 GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# 7. 开始训练!
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
```

---

## 📦 安装的内容

新配置会在虚拟环境中安装:

```toml
tensorflow==2.15.0                    # TensorFlow GPU 版本
nvidia-cudnn-cu12==9.1.0.70          # cuDNN 9.1 for CUDA 12
nvidia-cuda-runtime-cu12==12.3.101   # CUDA Runtime 12.3
nvidia-cublas-cu12==12.3.4.1         # cuBLAS
nvidia-cufft-cu12==11.0.12.1         # cuFFT
nvidia-curand-cu12==10.3.4.107       # cuRAND
nvidia-cusolver-cu12==11.5.4.101     # cuSOLVER
nvidia-cusparse-cu12==12.2.0.103     # cuSPARSE
```

这些包总大小约 **~2.5GB**,但会完全隔离在虚拟环境中!

---

## ✅ 验证成功

成功后应该看到:

```bash
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

$ uv pip list | grep nvidia
nvidia-cublas-cu12        12.3.4.1
nvidia-cuda-runtime-cu12  12.3.101
nvidia-cudnn-cu12         9.1.0.70
nvidia-cufft-cu12         11.0.12.1
nvidia-curand-cu12        10.3.4.107
nvidia-cusolver-cu12      11.5.4.101
nvidia-cusparse-cu12      12.2.0.103

$ python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
Using GPU 0 only.
GPU memory growth enabled for GPU 0
[训练开始...]
```

---

## 🔧 故障排除

### 问题 1: 仍然报 CUDA 错误

临时清除系统 CUDA 路径:
```bash
unset LD_LIBRARY_PATH
unset CUDA_HOME
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

如果这样能工作,在 `.env` 中添加:
```bash
unset LD_LIBRARY_PATH
unset CUDA_HOME
```

### 问题 2: 下载很慢

CUDA 包比较大(~2.5GB),下载需要时间。确保:
```bash
# 检查代理
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 如需代理,加载 .env
source .env
```

### 问题 3: 空间不足

检查磁盘空间:
```bash
df -h ~
```

需要至少 5GB 可用空间。

---

## 📊 优势

| 特性 | 系统 CUDA | 虚拟环境 CUDA |
|------|----------|--------------|
| 版本控制 | ❌ 全局版本 | ✅ 项目隔离 |
| 兼容性 | ❌ 可能不匹配 | ✅ 完美匹配 |
| 权限需求 | ❌ 需要 root | ✅ 用户级别 |
| 多项目 | ❌ 冲突风险 | ✅ 各自独立 |
| 可重现 | ❌ 依赖系统 | ✅ 完全可重现 |
| 与 PyTorch 一致 | ❌ 不同方式 | ✅ 统一方式 |

---

## 🎯 为什么不用 `tensorflow[and-cuda]`?

`tensorflow[and-cuda]==2.15.0` 有依赖问题:
```
tensorflow[and-cuda]==2.15.0 depends on tensorrt-libs==8.6.1
但 tensorrt-libs==8.6.1 在 PyPI 上不存在!
```

所以我们使用:
- ✅ `tensorflow==2.15.0` (标准版)
- ✅ 手动添加 NVIDIA CUDA 包
- ✅ 效果一样,但更可靠!

---

## 🚀 下一步

成功后,你可以:

1. **训练模型**:
   ```bash
   python scripts/train_neural_rx.py -config_name nrx_large
   ```

2. **评估模型**:
   ```bash
   python scripts/evaluate.py -config_name nrx_large
   ```

3. **查看进度**:
   ```bash
   tensorboard --logdir logs/
   ```

4. **运行 Jupyter**:
   ```bash
   jupyter notebook notebooks/jumpstart_tutorial.ipynb
   ```

---

## 📝 技术说明

### CUDA 包说明

- **nvidia-cudnn-cu12**: 深度学习加速库
- **nvidia-cuda-runtime-cu12**: CUDA 运行时库
- **nvidia-cublas-cu12**: 线性代数库(矩阵运算)
- **nvidia-cufft-cu12**: 快速傅里叶变换
- **nvidia-curand-cu12**: 随机数生成
- **nvidia-cusolver-cu12**: 线性系统求解
- **nvidia-cusparse-cu12**: 稀疏矩阵运算

这些包提供了 TensorFlow GPU 运算所需的全部 CUDA 功能!

---

祝训练顺利! 🎉

有问题随时问!
