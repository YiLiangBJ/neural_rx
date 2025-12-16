# 🚀 Neural RX 快速开始指南 - 最终版本

**恭喜!这是经过完整测试和优化的最终配置指南。**

---

## ✅ 系统要求

- **操作系统**: Linux (推荐 Ubuntu 22.04 LTS)
- **Python**: 3.10 (UV 会自动管理)
- **硬件**: 
  - CPU: 推荐 8 核以上
  - GPU: NVIDIA GPU + 驱动支持 CUDA 12.x (推荐)
  - 内存: 16GB 以上
  - 磁盘: 10GB 可用空间

---

## 📦 第一次安装

### 1. 克隆项目

```bash
git clone https://github.com/YiLiangBJ/neural_rx.git
cd neural_rx
```

### 2. 安装 UV 包管理器(如果还没有)

```bash
# 方式 1: 使用 pip
pip install uv

# 方式 2: 使用官方脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 配置代理(可选,仅在防火墙后需要)

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 添加你的代理
nano .env

# 加载环境
source .env
```

### 4. 安装依赖

**GPU 环境(推荐)**:
```bash
uv sync --extra gpu
```

**CPU 环境**:
```bash
uv sync --extra cpu
```

这会:
- ✅ 自动下载 Python 3.10(如需要)
- ✅ 创建虚拟环境在 `.venv/`
- ✅ 安装 TensorFlow 2.15.0
- ✅ 安装 Sionna 0.18.0
- ✅ 安装 CUDA 12.2 + cuDNN 8.9 (GPU 环境)
- ✅ 安装所有依赖包

**预计安装时间**: 5-10 分钟(取决于网络速度)

---

## 🎯 每次使用

### GPU 环境(一键激活)

```bash
cd ~/neural_rx
source activate_gpu.sh
```

这会:
- ✅ 激活虚拟环境
- ✅ 设置 CUDA 库路径
- ✅ 抑制 TensorFlow 警告
- ✅ 验证 GPU 可用性

### CPU 环境

```bash
cd ~/neural_rx
source .venv/bin/activate
```

---

## 🧪 验证安装

### 快速验证

```bash
# GPU 环境
source activate_gpu.sh

# 应该看到:
# ✅ 检测到 1 个 GPU
#    - /physical_device:GPU:0
```

### 完整验证

```bash
python verify_gpu.py
```

**预期输出(GPU 环境)**:
```
============================================================
检查 TensorFlow GPU 支持
============================================================
✅ TensorFlow 版本: 2.15.0
✅ GPU 可用: True
   检测到 1 块 GPU:
   - GPU 0: /physical_device:GPU:0
   
   CUDA 版本: 12.2
   cuDNN 版本: 8.9
   
   ✅ GPU 计算测试成功

============================================================
验证总结
============================================================
✅ 系统: Linux 5.15.0-xxx-generic
✅ CPU: 64 核心 / 128 线程
✅ 内存: 219.97GB
✅ TensorFlow GPU 检测成功
```

---

## 🏃 运行训练

### 查看可用配置

```bash
ls config/*.cfg
```

常用配置:
- `nrx_large.cfg` - 大型神经接收器
- `nrx_rt.cfg` - 实时推理优化版本
- `e2e_baseline.cfg` - 端到端基线
- `nrx_site_specific.cfg` - 站点特定训练

### 训练模型

```bash
# 训练 NRX Large 模型
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0

# 使用多个 GPU
python scripts/train_neural_rx.py -config_name nrx_large -gpu 1

# 调试模式
python scripts/train_neural_rx.py -config_name nrx_large -gpu 0 -debug
```

**预期输出**:
```
Using GPU 0 only.
GPU memory growth enabled for GPU 0

System Parameters:
  - Carrier frequency: 3.5 GHz
  - Bandwidth: 100 MHz
  - Users: 1-8
  - MCS: QPSK to 256-QAM
  
Training started...
Epoch 1/100: loss=0.523, BLER=0.123
...
```

### 评估模型

```bash
# 评估训练好的模型
python scripts/evaluate.py -config_name nrx_large -gpu 0

# 只评估神经网络(不评估基线)
python scripts/evaluate.py -config_name nrx_large -gpu 0 -eval_nrx_only

# 限制目标误块率
python scripts/evaluate.py -config_name nrx_large -gpu 0 -target_bler 0.001
```

---

## 📊 查看结果

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 在浏览器打开: http://localhost:6006
```

### 结果文件

训练和评估结果保存在:
```
results/
  ├── nrx_large_results/        # 评估结果
  │   ├── bler_vs_ebno.pkl      # BLER vs EbNo 数据
  │   └── throughput.pkl        # 吞吐量数据
  └── ...

weights/
  ├── nrx_large_weights/        # 模型权重
  │   ├── checkpoint            # TensorFlow checkpoint
  │   └── ...
  └── ...
```

---

## 📓 运行 Jupyter Notebooks

```bash
# 启动 Jupyter
jupyter notebook

# 或指定笔记本
jupyter notebook notebooks/jumpstart_tutorial.ipynb
```

推荐笔记本:
- `jumpstart_tutorial.ipynb` - 入门教程
- `nrx_architecture.ipynb` - NRX 架构详解
- `plot_results.ipynb` - 结果可视化
- `real_time_nrx.ipynb` - 实时推理
- `site_specific_neural_receivers.ipynb` - 站点特定训练

---

## 🔧 常见问题

### Q1: GPU 未检测到

**症状**:
```
❌ GPU 可用: False
```

**解决**:
```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi

# 2. 重新激活环境
source activate_gpu.sh

# 3. 验证
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Q2: 内存不足

**症状**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**解决**:
```bash
# 方法 1: 使用较小的配置
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0

# 方法 2: 在脚本中启用内存增长(已默认启用)
# GPU memory growth 会按需分配内存

# 方法 3: 减少批量大小(修改配置文件)
```

### Q3: 训练速度慢

**检查**:
```bash
# 1. 确认使用 GPU
nvidia-smi  # 应该看到 Python 进程占用 GPU

# 2. 检查 XLA 编译
# 首次运行会编译,后续会更快

# 3. 使用实时配置
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0
```

### Q4: 依赖冲突

**解决**:
```bash
# 清除所有缓存
rm -rf .venv uv.lock ~/.cache/uv

# 重新安装
uv sync --extra gpu
```

---

## 📚 进阶使用

### 导出 ONNX 模型

```bash
python scripts/export_onnx.py -config_name nrx_large
```

### 计算协方差矩阵

```bash
python scripts/compute_cov_mat.py -config_name nrx_site_specific
```

### 修改配置

复制并编辑配置文件:
```bash
cp config/nrx_large.cfg config/my_config.cfg
nano config/my_config.cfg

# 使用自定义配置
python scripts/train_neural_rx.py -config_name my_config -gpu 0
```

---

## 🎓 学习资源

### 官方文档

- [Sionna 文档](https://nvlabs.github.io/sionna/)
- [TensorFlow GPU 支持](https://www.tensorflow.org/install/gpu)
- [Neural Receiver 博客](https://developer.nvidia.com/blog/towards-environment-specific-base-stations-ai-ml-driven-neural-5g-nr-multi-user-mimo-receiver/)

### 项目文档

- [README.md](../README.md) - 项目概览
- [SETUP.md](../SETUP.md) - 详细安装指南
- [CONFIGURATION_SUMMARY.md](../CONFIGURATION_SUMMARY.md) - 配置说明
- [docs/GPU_SETUP_FINAL.md](GPU_SETUP_FINAL.md) - GPU 配置详解
- [docs/TENSORFLOW_GPU_FIX.md](TENSORFLOW_GPU_FIX.md) - GPU 问题排查

### 论文

1. **Neural Receiver Design**: [arXiv:2021.xxxxx](https://arxiv.org/abs/xxx)
2. **Pilotless Communications**: [arXiv:2009.05261](https://arxiv.org/abs/2009.05261)
3. **Site-Specific Training**: [IEEE Paper](https://ieeexplore.ieee.org/document/xxx)

---

## 🛠️ 开发工作流

### 日常开发

```bash
# 1. 激活环境
cd ~/neural_rx
source activate_gpu.sh

# 2. 修改代码
nano utils/neural_rx.py

# 3. 测试
python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0 -debug

# 4. 评估
python scripts/evaluate.py -config_name nrx_rt -gpu 0

# 5. 提交(如果满意)
git add .
git commit -m "Improved neural receiver architecture"
git push
```

### 实验管理

```bash
# 创建实验分支
git checkout -b experiment/new-architecture

# 修改配置
cp config/nrx_large.cfg config/nrx_new_arch.cfg

# 训练
python scripts/train_neural_rx.py -config_name nrx_new_arch -gpu 0

# 对比结果
python notebooks/plot_results.ipynb

# 如果效果好,合并到 main
git checkout main
git merge experiment/new-architecture
```

---

## 📊 性能优化

### GPU 利用率

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 应该看到:
# GPU-Util: 95-100%  ✅ 很好
# GPU-Util: 50-70%   ⚠️  考虑增加批量大小
# GPU-Util: <30%     ❌ 检查是否使用 CPU
```

### 训练加速

1. **使用 XLA 编译**(已默认启用):
   ```python
   # 在配置文件中
   xla = True
   ```

2. **混合精度训练**:
   ```python
   # 在训练脚本中添加
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

3. **多 GPU 训练**(实验性):
   ```bash
   # 修改 train_neural_rx.py 中的 distribute 变量
   distribute = "all"
   ```

---

## 🎉 成功标志

当你看到以下输出时,说明一切正常:

```bash
$ source activate_gpu.sh
🚀 激活 Neural RX GPU 环境...

✅ 虚拟环境: /home/xxx/neural_rx/.venv
✅ CUDA 库路径已设置
✅ TensorFlow 警告已抑制

🔍 验证 GPU 可用性...
✅ 检测到 1 个 GPU
   - /physical_device:GPU:0

📋 可用命令:
   python verify_gpu.py                                  # 完整系统验证
   python scripts/train_neural_rx.py -config_name <cfg>  # 训练模型
   python scripts/evaluate.py -config_name <cfg>         # 评估模型

$ python scripts/train_neural_rx.py -config_name nrx_large -gpu 0
Using GPU 0 only.
GPU memory growth enabled for GPU 0
[训练开始...]
```

---

## 💬 获取帮助

- **GitHub Issues**: [https://github.com/YiLiangBJ/neural_rx/issues](https://github.com/YiLiangBJ/neural_rx/issues)
- **文档**: 查看 `docs/` 目录
- **Notebooks**: 参考 `notebooks/` 示例

---

## ✨ 下一步

1. ✅ 运行 `jumpstart_tutorial.ipynb` 熟悉基本概念
2. ✅ 训练第一个模型: `python scripts/train_neural_rx.py -config_name nrx_rt -gpu 0`
3. ✅ 评估模型性能: `python scripts/evaluate.py -config_name nrx_rt -gpu 0`
4. ✅ 查看 TensorBoard: `tensorboard --logdir logs/`
5. ✅ 尝试不同配置和参数
6. ✅ 发表你的研究成果! 📄

---

**祝训练顺利!** 🚀🎊

如有问题,请参考文档或提交 Issue。
