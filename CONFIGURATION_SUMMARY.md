# ✅ Neural RX 环境配置完成!

## 📝 修改摘要

已完成以下配置,严格按照官方 README 要求:

### 1. `pyproject.toml` - 依赖管理配置
- ✅ Python 版本: **3.10** (官方推荐,严格限制 `>=3.10,<3.11`)
- ✅ **自动 Python 版本管理**: 配置了 `python-version = "3.10"` + `.python-version` 文件
- ✅ 三种依赖组:
  - **windows-cpu**: TF 2.10.1 + Sionna 0.14 (Windows 限制,TF 2.15+ 不支持 Windows)
  - **linux-cpu**: TF 2.15 + Sionna 0.18 + Mitsuba (官方推荐配置)
  - **linux-gpu**: TF 2.15 + Sionna 0.18 + TensorRT 9.6+ + Mitsuba (完整配置)

### 2. `.python-version` - Python 版本锁定
- ✅ 指定 Python 3.10
- ✅ `uv sync` 会自动查找并使用 Python 3.10

### 3. `.env` - 环境变量配置
- ✅ Intel 代理配置
- ✅ Python 3.10 路径 (自动检测 Windows/Linux)
- ✅ TensorFlow 日志级别设置
- ✅ CUDA 设备配置(可选)

### 4. `verify_gpu.py` - 系统能力验证脚本 (新增)
- ✅ 从 `VerifyGPU_CUDA_cuDNN.ipynb` 转换而来并大幅增强
- ✅ **完整系统信息**: CPU、内存、磁盘、操作系统
- ✅ **GPU 检测**: TensorFlow 和 PyTorch 的 GPU 支持
- ✅ **CUDA/cuDNN**: 版本信息和性能测试
- ✅ **性能评估**: 自动评估机器适合的工作负载
- ✅ 可以直接运行: `python verify_gpu.py`

### 5. 文档
- ✅ `SETUP.md` - 完整安装指南
- ✅ `QUICKSTART.md` - 快速开始
- ✅ `.env.example` - 环境配置模板

---

## 🚀 下一步操作

### 在你的 Windows 机器上 (当前环境):

1. **确保有 Python 3.10**:
   ```bash
   python --version  # 应该显示 Python 3.10.x
   ```
   
   如果没有,从 https://www.python.org/downloads/ 下载安装 Python 3.10

2. **删除旧虚拟环境**:
   ```bash
   rm -rf .venv
   rm -f uv.lock  # 删除旧的锁定文件
   ```

3. **创建新环境 (uv 会自动使用 Python 3.10)**:
   ```bash
   source .env  # 加载代理配置
   uv sync --extra windows-cpu
   source .venv/Scripts/activate
   ```

4. **验证安装**:
   ```bash
   python verify_gpu.py  # 运行 GPU 验证脚本
   ```

---

### 在 Linux CPU 机器上:

```bash
source .env  # 如需代理
rm -rf .venv
rm -f uv.lock
uv sync --extra linux-cpu  # uv 会自动使用 Python 3.10
source .venv/bin/activate
python verify_gpu.py  # 验证环境
```

---

### 在 Linux GPU 机器上:

```bash
source .env  # 如需代理
rm -rf .venv
rm -f uv.lock
uv sync --extra linux-gpu  # uv 会自动使用 Python 3.10
source .venv/bin/activate
python verify_gpu.py  # 验证 GPU 和 CUDA
```

---

## 📦 依赖版本对照

| 环境 | TensorFlow | Sionna | Mitsuba | TensorRT | 备注 |
|------|-----------|--------|---------|----------|------|
| **Windows CPU** | 2.10.1 (CPU) | 0.14.0 | ❌ | ❌ | Windows 限制 |
| **Linux CPU** | 2.15.0 (CPU) | 0.18.0 | 3.5.2 | ❌ | 官方推荐 |
| **Linux GPU** | 2.15.0 (CUDA) | 0.18.0 | 3.5.2 | 9.6+ | 完整功能 |

---

## ⚠️ 重要提示

1. **Python 版本**: 必须使用 **3.10** (官方推荐)
   - ✅ `.python-version` 文件已配置,`uv sync` 会自动查找 Python 3.10
   
2. **ONNX 版本**: 必须使用 **1.14** (1.15 有已知 bug)

3. **Windows vs Linux 版本差异** (这是故意的!):
   - **Windows**: TF 2.10.1 + Sionna 0.14 (TF 2.15+ 不支持 Windows)
   - **Linux**: TF 2.15 + Sionna 0.18 (官方推荐,功能完整)

4. **Windows 限制**: Mitsuba 和 TensorRT 仅支持 Linux

5. **代理设置**: Intel 内网用户务必先 `source .env`

6. **GPU 验证**: 运行 `python verify_gpu.py` 检查环境

---

## 🎯 使用命令速查

```bash
# 查看当前安装的包
uv pip list

# 更新所有包
uv sync --upgrade --extra <windows-cpu|linux-cpu|linux-gpu>

# 添加开发工具
uv sync --extra linux-gpu --extra dev

# 检查环境
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

## 📖 参考文档

- 完整说明: `SETUP.md`
- 快速开始: `QUICKSTART.md`
- 环境配置: `.env.example`

祝使用顺利! 🎉
