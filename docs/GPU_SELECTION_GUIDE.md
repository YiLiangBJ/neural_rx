# 🖥️ GPU 选择和配置指南

## 概述

Neural RX 支持灵活的 GPU 配置,可以选择:
- **单个 GPU** (GPU 0, 1, 2, ...)
- **所有 GPU** (多 GPU 分布式训练)
- **CPU** (无 GPU 环境)

---

## 🎯 GPU 选择选项

### 1. 使用特定 GPU

选择单个 GPU 进行训练或评估:

```bash
# 使用 GPU 0
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 使用 GPU 1
python scripts/train_neural_rx.py -config_name nrx_large -gpu 1

# 使用 GPU 2
python scripts/train_neural_rx.py -config_name nrx_large -gpu 2
```

**特点**:
- ✅ 简单直接
- ✅ 完全控制使用哪个 GPU
- ✅ 避免 GPU 冲突(多人共享服务器)
- ✅ 内存增长模式自动启用

**适用场景**:
- 单 GPU 机器
- 多人共享服务器(每人用不同 GPU)
- 测试特定 GPU 性能
- 避免占用所有 GPU

---

### 2. 使用所有 GPU (分布式训练)

自动使用所有可用 GPU:

```bash
# 使用所有 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu all
```

**特点**:
- ✅ 自动分布式训练
- ✅ 充分利用硬件资源
- ✅ 训练速度成倍提升
- ⚠️ 需要足够的显存(每个 GPU)
- ⚠️ 批量大小会自动分配到各 GPU

**分布式策略**:
- 自动使用 `tf.distribute.MirroredStrategy`
- 数据自动分片到各 GPU
- 梯度自动聚合
- 权重同步更新

**适用场景**:
- 独占服务器
- 大规模训练
- 追求最快训练速度
- 有多个 GPU 可用

---

### 3. 使用 CPU (无 GPU)

强制使用 CPU:

```bash
# 使用 CPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu cpu
```

**特点**:
- ✅ 无需 GPU
- ✅ 适合调试
- ⚠️ 训练速度**非常慢**
- ⚠️ 只推荐用于小规模测试

**适用场景**:
- 没有 GPU 的机器
- 快速验证代码逻辑
- CPU 性能测试
- 开发环境调试

---

## 📊 查看可用 GPU

### 方法 1: 使用 nvidia-smi

```bash
nvidia-smi

# 输出示例:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA A10          Off  | 00000000:00:1E.0 Off |                  Off |
# |   1  NVIDIA A10          Off  | 00000000:00:1F.0 Off |                  Off |
# +-------------------------------+----------------------+----------------------+
```

### 方法 2: 使用 Python

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"可用 GPU 数量: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu.name}")
```

### 方法 3: 使用 verify_gpu.py

```bash
python verify_gpu.py

# 输出会显示所有可用 GPU
```

---

## 🚀 使用示例

### 场景 1: 单 GPU 机器

```bash
# 只有一个 GPU,使用 GPU 0
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 或者使用 all (效果相同)
python scripts/train_neural_rx.py -config_name nrx_large -gpu all
```

---

### 场景 2: 双 GPU 机器

```bash
# 查看 GPU 状态
nvidia-smi

# 选项 A: 只用 GPU 0
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 选项 B: 只用 GPU 1
python scripts/train_neural_rx.py -config_name nrx_large -gpu 1

# 选项 C: 用所有 GPU (推荐,最快)
python scripts/train_neural_rx.py -config_name nrx_large -gpu all
```

---

### 场景 3: 多人共享服务器 (4 GPU)

```bash
# 查看 GPU 使用情况
nvidia-smi

# 假设 GPU 0 和 1 被占用,使用 GPU 2
python scripts/train_neural_rx.py -config_name nrx_large -gpu 2

# 或者使用 GPU 3
python scripts/train_neural_rx.py -config_name nrx_large -gpu 3
```

**最佳实践**:
1. 先用 `nvidia-smi` 查看哪些 GPU 空闲
2. 选择空闲的 GPU
3. 与其他用户协调使用

---

### 场景 4: 没有 GPU 的机器

```bash
# 使用 CPU (仅用于测试)
python scripts/train_neural_rx.py -config_name nrx_rt -gpu cpu

# 建议使用最小配置 nrx_rt
```

---

## 🔍 训练输出示例

### 使用 GPU 0

```bash
$ python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

🎯 使用 GPU 0: /physical_device:GPU:0
   已启用内存增长模式

======================================================================
🚀 开始训练
======================================================================
📋 配置: nrx_large
🏷️  标签: nrx_large
🖥️  计算设备: GPU 0
💾 权重路径: /path/to/weights/nrx_large_weights
...
```

---

### 使用所有 GPU

```bash
$ python scripts/train_neural_rx.py -config_name nrx_large -gpu all

📊 使用所有 2 个 GPU 进行分布式训练
   策略: MirroredStrategy
   GPU 列表: ['/physical_device:GPU:0', '/physical_device:GPU:1']

🔧 在分布式策略中创建模型...
✅ 分布式模型创建完成

======================================================================
🚀 开始训练
======================================================================
📋 配置: nrx_large
🏷️  标签: nrx_large
🖥️  计算设备: 2 个 GPU (分布式训练)
   策略: MirroredStrategy
...
```

---

### 使用 CPU

```bash
$ python scripts/train_neural_rx.py -config_name nrx_large -gpu cpu

🖥️  使用 CPU 训练 (所有 GPU 已禁用)
   ⚠️  警告: CPU 训练会非常慢!

======================================================================
🚀 开始训练
======================================================================
📋 配置: nrx_large
🏷️  标签: nrx_large
🖥️  计算设备: CPU
...
```

---

## 📈 性能对比

### 训练速度 (相对于单 GPU)

| 配置 | 相对速度 | 推荐场景 |
|------|---------|---------|
| **单 GPU** | 1x (基准) | 标准训练 |
| **双 GPU (all)** | ~1.8x | 大规模训练 |
| **四 GPU (all)** | ~3.5x | 超大规模 |
| **CPU** | 0.01x (慢 100 倍) | 仅测试 |

*实际加速比取决于模型大小和通信开销*

---

## 🛠️ 高级配置

### 1. 限制 GPU 显存使用

```python
# 在训练脚本中添加
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 限制每个 GPU 使用 8GB
    for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
        )
```

---

### 2. 使用环境变量选择 GPU

```bash
# 方法 1: 使用 CUDA_VISIBLE_DEVICES (已废弃,不推荐)
# CUDA_VISIBLE_DEVICES=1 python scripts/train_neural_rx.py ...

# 方法 2: 使用 -gpu 参数 (推荐)
python scripts/train_neural_rx.py -config_name nrx_large -gpu 1
```

---

### 3. 监控 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi dmon -s pucvmet -d 1

# 查看特定 GPU
nvidia-smi -i 0
```

---

## 🔧 故障排除

### Q1: 提示 GPU 不存在

```bash
❌ GPU 2 不存在! 可用 GPU 数量: 2
   可用选项: 0-1, "all", 或 "cpu"
```

**解决**: 检查可用 GPU 数量,使用正确的 GPU 编号

```bash
nvidia-smi  # 查看有几个 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0  # 使用 0 或 1
```

---

### Q2: 显存不足 (OOM)

```
ResourceExhaustedError: OOM when allocating tensor
```

**解决方案**:

```bash
# 方法 1: 使用单个 GPU (不用 all)
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 方法 2: 使用更小的配置
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0

# 方法 3: 减小 batch size (修改配置文件)
```

---

### Q3: 多 GPU 训练没有加速

**可能原因**:
1. 批量大小太小(通信开销大于计算)
2. 模型太小(分布式开销占比大)
3. 数据加载是瓶颈

**解决**:
- 增大 batch size
- 使用更大的模型
- 优化数据管道

---

### Q4: GPU 被其他进程占用

```bash
# 查看 GPU 占用
nvidia-smi

# 如果看到其他进程,选择空闲的 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu 1  # 使用 GPU 1
```

---

## 💡 最佳实践

### 1. 开发阶段

```bash
# 使用单个 GPU + 小配置 + debug 模式
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0 -debug
```

### 2. 测试阶段

```bash
# 使用单个 GPU + 小配置
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0
```

### 3. 生产训练

```bash
# 使用所有 GPU + 大配置
python scripts/train_neural_rx.py -config_name nrx_large -gpu all
```

### 4. 共享服务器

```bash
# 1. 检查 GPU 使用情况
nvidia-smi

# 2. 选择空闲的 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu 2  # 假设 GPU 2 空闲

# 3. 与团队协调使用
```

---

## 📚 相关命令

### 训练

```bash
# 单 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 多 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu all

# CPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu cpu
```

### 评估

```bash
# 单 GPU
python scripts/evaluate.py -config_name nrx_large -gpu 0

# 多 GPU
python scripts/evaluate.py -config_name nrx_large -gpu all

# CPU
python scripts/evaluate.py -config_name nrx_large -gpu cpu
```

### 监控

```bash
# GPU 使用率
watch -n 1 nvidia-smi

# GPU 详细信息
nvidia-smi -i 0 -q

# TensorBoard
tensorboard --logdir logs/
```

---

## 总结

| 场景 | 推荐配置 | 命令 |
|------|---------|------|
| 单 GPU 机器 | GPU 0 或 all | `-gpu 0` 或 `-gpu all` |
| 多 GPU 机器(独占) | 所有 GPU | `-gpu all` |
| 多 GPU 机器(共享) | 指定空闲 GPU | `-gpu 1` |
| 没有 GPU | CPU | `-gpu cpu` |
| 开发调试 | 单 GPU + debug | `-gpu 0 -debug` |
| 生产训练 | 所有 GPU | `-gpu all` |

**记住**: 合理利用 GPU 资源,与他人协调使用,避免冲突! ✨
