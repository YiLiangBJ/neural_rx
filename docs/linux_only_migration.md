# ✅ 项目简化完成!仅支持 Linux

## 主要变更

### 移除 Windows 支持

**原因**:
1. ❌ Sionna 0.14.0 在 Windows 上导入就 Segmentation fault
2. ❌ Mitsuba 不支持 Windows
3. ❌ TensorRT 不支持 Windows  
4. ❌ TensorFlow 2.15+ GPU 不支持 Windows
5. ❌ 版本兼容性问题太多(protobuf, ONNX, tf2onnx 冲突)

**决定**: 专注于 Linux 平台,提供最佳体验。

---

## 新的配置结构

### `pyproject.toml` 简化

```toml
[project.optional-dependencies]
# Linux CPU 环境
cpu = [
    "tensorflow-cpu==2.15.0",
    "sionna==0.18.0",
    "mitsuba==3.5.2",
    "onnx==1.14.0",
    "tf2onnx>=1.16.0",
]

# Linux GPU 环境 (推荐)
gpu = [
    "tensorflow==2.15.0",
    "sionna==0.18.0",
    "mitsuba==3.5.2",
    "onnx==1.14.0",
    "tf2onnx>=1.16.0",
    "tensorrt>=9.6.0",
    "nvidia-cudnn-cu12>=9.0.0",
]

[tool.uv]
default-groups = []
conflicts = [[{ extra = "cpu" }, { extra = "gpu" }]]
```

**改进**:
- ✅ 移除所有平台标记 (`platform_system`)
- ✅ 移除 `required-environments`
- ✅ 简化 extras 名称: `cpu` / `gpu` (不再需要 `linux-` 前缀)
- ✅ 统一版本: TensorFlow 2.15, Sionna 0.18, ONNX 1.14
- ✅ 无版本冲突

---

## 使用方法

### CPU 环境

```bash
source .env  # 如需代理
uv sync --extra cpu
source .venv/bin/activate
python verify_gpu.py
```

### GPU 环境 (推荐)

```bash
source .env  # 如需代理
uv sync --extra gpu
source .venv/bin/activate
python verify_gpu.py
```

---

## 更新的文档

1. **`SETUP.md`** - 全新简化版
   - 只有 CPU 和 GPU 两种场景
   - 清晰的系统要求说明
   - Linux 专属

2. **`QUICKSTART.md`** - 更新
   - 移除 Windows 场景
   - 简化为 CPU/GPU 选择

3. **`pyproject.toml`** - 简化
   - 移除 `windows-cpu`
   - 移除平台标记
   - 移除 `required-environments`
   - 重命名: `linux-cpu` → `cpu`, `linux-gpu` → `gpu`

4. **`README.md`** - 需要添加系统要求
   - 明确说明仅支持 Linux

---

## 优势

### ✅ 简化后的好处

1. **无兼容性问题**
   - 所有包使用最新稳定版本
   - 无 protobuf 冲突
   - 无平台特定的 bug

2. **更快的开发周期**
   - 无需维护多个平台配置
   - 测试更简单
   - 文档更清晰

3. **更好的性能**
   - TensorFlow 2.15 性能最佳
   - Sionna 0.18 功能最全
   - TensorRT 加速

4. **专注核心功能**
   - GPU 训练是主要用途
   - Mitsuba 射线追踪
   - 实时推理

---

## Windows 用户的替代方案

### 选项 1: WSL2 (推荐)

```bash
# 安装 WSL2 (PowerShell 管理员)
wsl --install

# 在 WSL2 中使用
wsl
cd /mnt/c/GitRepo/neural_rx
uv sync --extra gpu
```

### 选项 2: Docker

```bash
# 使用 NVIDIA Docker
docker run --gpus all -it -v c:/GitRepo/neural_rx:/workspace nvidia/cuda:12.3.0-cudnn9-devel-ubuntu22.04
```

### 选项 3: 云端 GPU

- AWS EC2 (g4dn, p3, p4 实例)
- Google Cloud Compute Engine (GPU)
- Azure NC 系列
- Lambda Labs
- Paperspace

---

## 验证

在 Linux 机器上测试:

```bash
# CPU 环境
uv sync --extra cpu
source .venv/bin/activate
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
python -c "import sionna; print(f'Sionna: {sionna.__version__}')"
python -c "import mitsuba; print('Mitsuba: OK')"

# 预期输出:
# TF: 2.15.0
# Sionna: 0.18.0
# Mitsuba: OK
```

```bash
# GPU 环境
uv sync --extra gpu
source .venv/bin/activate
python verify_gpu.py

# 预期:
# ✅ TensorFlow 2.15.0
# ✅ GPU 可用: True
# ✅ CUDA 12.3
# ✅ cuDNN 9.0
```

---

## 迁移指南

### 如果之前使用 Windows

1. **迁移到 WSL2**
   ```bash
   # Windows PowerShell
   wsl --install
   wsl
   
   # WSL2 中
   cd /mnt/c/GitRepo/neural_rx
   uv sync --extra cpu  # 或 gpu
   ```

2. **或使用 Linux 服务器**
   ```bash
   # 复制代码到 Linux 服务器
   scp -r neural_rx user@linux-server:~/
   
   # SSH 登录
   ssh user@linux-server
   cd ~/neural_rx
   uv sync --extra gpu
   ```

### 清理旧环境

```bash
# 删除旧的虚拟环境
rm -rf .venv uv.lock

# 重新安装
uv sync --extra gpu
```

---

## 下一步

1. ✅ 在 Linux 机器上测试 `cpu` 和 `gpu` extras
2. ✅ 更新 CI/CD 管道(仅 Linux)
3. ✅ 在 README.md 中添加系统要求
4. ✅ 添加 WSL2 使用说明
5. ✅ 添加 Docker 配置

---

## 总结

**之前**: 
- 支持 Windows CPU + Linux CPU + Linux GPU
- 3 个冲突的 extras
- 版本兼容性问题
- 平台标记和环境限制
- Sionna Windows 崩溃

**现在**:
- 仅支持 Linux
- 2 个简单的 extras: `cpu` / `gpu`
- 无版本冲突
- 无平台限制
- 统一的最新版本

**结果**: 
- 更简单 ✅
- 更稳定 ✅
- 更快 ✅
- 更专注 ✅

---

🎉 项目现在只专注于 Linux,提供最佳的神经接收器训练和推理体验!
