# 🐛 Neural RX 调试模式指南

## 概述

Neural RX 提供了多种调试选项,让你可以在**快速开发调试**和**高性能训练**之间灵活选择。

---

## 🎯 调试选项

### 1. 完整调试模式 (`-debug`)

**用途**: 深度调试,逐步执行,设置断点

**启用**:
```bash
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0 -debug
```

**效果**:
- ✅ **Eager Execution**: 启用 TensorFlow eager 模式
- ✅ **禁用 XLA**: 无编译等待
- ✅ **可设置断点**: 可以用 `pdb` 或 IDE 调试器
- ✅ **单独日志**: 保存到 `logs/debug/`
- ⚠️ **速度慢**: 比正常训练慢 **10-100倍**

**适用场景**:
- 调试新功能
- 检查中间变量
- 追踪 bug
- 理解网络执行细节

---

### 2. 仅禁用 XLA (`--no-xla`)

**用途**: 快速启动,无需等待 XLA 编译

**启用**:
```bash
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0 --no-xla
```

**效果**:
- ✅ **无编译等待**: 立即开始训练
- ✅ **保留图模式**: 仍使用 `@tf.function`
- ⚠️ **速度较慢**: 比 XLA 慢 **2-5倍**
- ✅ **不能设置断点**: 因为还在图模式

**适用场景**:
- 快速验证配置
- 测试小规模数据
- 检查训练是否能运行
- 不想等待长时间 XLA 编译

---

### 3. 正常模式(默认)

**用途**: 生产训练,最佳性能

**启用**:
```bash
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
# 不加任何调试参数
```

**效果**:
- ✅ **XLA 编译**: 首次编译(可能需要 10-30 分钟)
- ✅ **最快速度**: 训练速度最快
- ✅ **图模式**: 使用 `@tf.function`
- ⚠️ **首次启动慢**: 需要等待 XLA 编译

**适用场景**:
- 正式训练
- 长时间运行
- 性能评测
- 生产部署

---

## 📊 性能对比

| 模式 | 启动时间 | 训练速度 | 可调试性 | 适用场景 |
|------|---------|---------|---------|---------|
| `-debug` | **快** (秒级) | **很慢** (1x) | ✅ **完全** | 深度调试 |
| `--no-xla` | **快** (秒级) | **较慢** (2-5x) | ❌ 部分 | 快速验证 |
| 正常模式 | **慢** (分钟级) | **最快** (10-100x) | ❌ 无 | 正式训练 |

*速度倍数相对于 debug 模式*

---

## 🔍 使用示例

### 场景 1: 调试新的损失函数

```bash
# 使用完整调试模式
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0 -debug

# 在代码中设置断点
# utils/neural_rx.py:
import pdb; pdb.set_trace()  # 在关键位置设置断点
```

**输出**:
```
🐛 调试模式已激活:
   - Eager execution: 启用 (可以设置断点)
   - XLA 编译: 禁用 (无编译等待)
   - 日志目录: logs/debug/
   ⚠️  注意: 调试模式会显著降低训练速度!
```

---

### 场景 2: 快速测试配置是否正确

```bash
# 禁用 XLA,快速启动
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0 --no-xla

# 训练几个迭代后 Ctrl+C 停止
# 确认配置正确后,用正常模式重新训练
```

**输出**:
```
⚡ XLA 编译已禁用
   ✅ 优点: 无编译等待,快速启动
   ⚠️  缺点: 训练速度较慢
```

---

### 场景 3: 正式训练(生产模式)

```bash
# 不加任何调试参数
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 第一次会看到:
# "Compiled cluster using XLA!" (等待 10-30 分钟)
# 之后训练速度非常快
```

---

## 🛠️ 高级调试技巧

### 1. 使用 TensorFlow Debugger (tfdbg)

```python
# 在训练脚本开头添加
import tensorflow as tf
tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
```

### 2. 打印中间张量

```python
# 在 debug 模式下可以直接打印
if args.debug:
    @tf.function
    def my_function(x):
        print("x =", x)  # 在 eager 模式下会打印
        return x * 2
```

### 3. 条件断点

```python
# 只在特定条件下断点
import pdb

def train_step(x):
    loss = compute_loss(x)
    if loss > 1.0:  # 损失异常大
        pdb.set_trace()  # 触发断点
    return loss
```

### 4. 性能分析

```bash
# 使用 TensorFlow Profiler
TF_CPP_MIN_LOG_LEVEL=0 python scripts/train_neural_rx.py \
    -config_name nrx_large -gpu 0 --no-xla

# 然后在 TensorBoard 查看性能
tensorboard --logdir logs/
```

---

## 📋 配置文件中的 XLA 设置

在配置文件(如 `config/nrx_large.cfg`)中:

```ini
[training]
xla = True  # 默认启用 XLA
```

**优先级**:
1. 命令行 `-debug` (最高优先级,强制禁用 XLA)
2. 命令行 `--no-xla` (禁用 XLA,保留图模式)
3. 配置文件 `xla = True/False` (默认设置)

---

## 💡 最佳实践

### 开发流程

```bash
# 1. 开发阶段: 使用 debug 模式
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0 -debug
# 修改代码,设置断点,理解逻辑

# 2. 验证阶段: 使用 --no-xla
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0 --no-xla
# 快速验证改动是否正确

# 3. 测试阶段: 使用较小的配置 + 正常模式
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0
# 完整测试,等待 XLA 编译

# 4. 生产阶段: 使用大配置 + 正常模式
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
# 正式训练,获得最佳性能
```

---

## 🔧 常见问题

### Q1: Debug 模式下为什么这么慢?

**A**: Debug 模式禁用了所有优化:
- 禁用 XLA 编译优化
- 禁用图模式自动优化
- 启用 eager execution(逐操作执行)

这是**正常的**,换来的是完整的调试能力。

---

### Q2: 可以部分启用 XLA 吗?

**A**: 可以!修改配置文件:

```ini
[training]
xla = True  # 训练时启用 XLA

[evaluation]
xla = False  # 评估时禁用 XLA
```

或者在代码中:
```python
@tf.function(jit_compile=False)  # 特定函数禁用 XLA
def my_function(x):
    return x * 2
```

---

### Q3: XLA 编译缓存在哪里?

**A**: XLA 缓存在内存中,进程结束后消失。但可以设置:

```bash
# 设置 XLA 缓存目录
export XLA_FLAGS="--xla_dump_to=/tmp/xla_cache"
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
```

---

### Q4: 如何查看 XLA 编译了什么?

**A**: 启用 XLA 日志:

```bash
# 详细 XLA 日志
TF_XLA_FLAGS="--tf_xla_clustering_debug" \
XLA_FLAGS="--xla_hlo_graph_dump_path=/tmp/xla_dumps" \
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
```

---

## 📚 参考资源

- [TensorFlow Debugging Guide](https://www.tensorflow.org/guide/debugging)
- [XLA Overview](https://www.tensorflow.org/xla)
- [Eager Execution](https://www.tensorflow.org/guide/eager)
- [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler)

---

## 总结

| 需求 | 使用模式 | 命令 |
|------|---------|------|
| 🐛 深度调试 | `-debug` | `python scripts/train_neural_rx.py ... -debug` |
| ⚡ 快速验证 | `--no-xla` | `python scripts/train_neural_rx.py ... --no-xla` |
| 🚀 正式训练 | 正常模式 | `python scripts/train_neural_rx.py ...` |

**记住**: 调试模式牺牲性能换取调试能力,生产模式牺牲启动时间换取运行性能。根据需求选择合适的模式! ✨
