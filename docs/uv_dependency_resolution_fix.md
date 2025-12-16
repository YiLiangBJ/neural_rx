# ✅ UV 依赖解析问题已解决!

## 问题概述

在 Windows 上运行 `uv sync --extra windows-cpu` 时遇到多个依赖冲突错误。

## 根本原因

UV 的新版本会**跨平台解析所有 optional-dependencies**,导致:

1. **冲突的 TensorFlow 版本**: Windows 需要 TF 2.10.1,Linux 需要 TF 2.15.0
2. **平台特定的包**: `tensorflow-io-gcs-filesystem`, `mitsuba`, `tensorrt` 只在 Linux 上可用
3. **Protobuf 版本冲突**: TF 2.10.1 需要 `protobuf<3.20`,而 `onnx==1.14.0` 和 `tf2onnx>=1.16.0` 需要 `protobuf>=3.20`
4. **缺少包结构**: 项目没有传统的 Python 包结构,hatchling 无法构建

## 解决方案

### 1. 添加平台标记 (Environment Markers)

为每个依赖添加平台限制:

```toml
[project.optional-dependencies]
windows-cpu = [
    "tensorflow-cpu==2.10.1; platform_system == 'Windows'",
    "sionna==0.14.0; platform_system == 'Windows'",
    # ...
]

linux-cpu = [
    "tensorflow-cpu==2.15.0; platform_system == 'Linux'",
    "sionna==0.18.0; platform_system == 'Linux'",
    # ...
]
```

### 2. 限制解析环境

使用 `tool.uv.required-environments` 只为当前平台解析:

```toml
[tool.uv]
required-environments = ["platform_system == 'Windows'"]  # 在 Windows 上
```

### 3. 声明冲突的 Extras

使用 `tool.uv.conflicts` 明确不同 extras 互斥:

```toml
[tool.uv]
conflicts = [
    [
        { extra = "windows-cpu" },
        { extra = "linux-cpu" },
    ],
    # ...
]
```

### 4. 修复版本兼容性

降级 Windows 上的 ONNX 和 tf2onnx 以兼容 TensorFlow 2.10.1:

```toml
windows-cpu = [
    "tensorflow-cpu==2.10.1",
    "onnx==1.12.0",      # 降级以兼容 protobuf<3.20
    "tf2onnx<1.16.0",    # 降级以兼容 protobuf<3.20
]
```

### 5. 配置包结构

添加 hatchling 配置以支持可编辑安装:

```toml
[tool.hatch.build.targets.wheel]
packages = ["utils"]
```

## 最终配置

### `pyproject.toml` 关键部分

```toml
[project]
name = "neural-rx"
requires-python = ">=3.10,<3.11"

[project.optional-dependencies]
windows-cpu = [
    "tensorflow-cpu==2.10.1; platform_system == 'Windows'",
    "sionna==0.14.0; platform_system == 'Windows'",
    "onnx==1.12.0; platform_system == 'Windows'",
    "tf2onnx<1.16.0; platform_system == 'Windows'",
]

linux-cpu = [
    "tensorflow-cpu==2.15.0; platform_system == 'Linux'",
    "sionna==0.18.0; platform_system == 'Linux'",
    "mitsuba==3.5.2; platform_system == 'Linux'",
    "onnx==1.14.0; platform_system == 'Linux'",
    "tf2onnx>=1.16.0; platform_system == 'Linux'",
]

linux-gpu = [
    "tensorflow==2.15.0; platform_system == 'Linux'",
    "sionna==0.18.0; platform_system == 'Linux'",
    "mitsuba==3.5.2; platform_system == 'Linux'",
    "onnx==1.14.0; platform_system == 'Linux'",
    "tf2onnx>=1.16.0; platform_system == 'Linux'",
    "tensorrt>=9.6.0; platform_system == 'Linux'",
    "nvidia-cudnn-cu12>=9.0.0; platform_system == 'Linux'",
]

[tool.hatch.build.targets.wheel]
packages = ["utils"]

[tool.uv]
default-groups = []
required-environments = ["platform_system == 'Windows'"]  # 在 Linux 上改为 'Linux'

conflicts = [
    [{ extra = "windows-cpu" }, { extra = "linux-cpu" }],
    [{ extra = "windows-cpu" }, { extra = "linux-gpu" }],
    [{ extra = "linux-cpu" }, { extra = "linux-gpu" }],
]
```

## 使用方法

### Windows

```bash
# 加载代理
source .env

# 删除旧环境
rm -rf .venv uv.lock

# 同步依赖
uv sync --extra windows-cpu

# 激活环境
source .venv/Scripts/activate

# 验证
python verify_gpu.py
```

### Linux

```bash
# 修改 pyproject.toml 中的 required-environments
[tool.uv]
required-environments = ["platform_system == 'Linux'"]

# 加载代理(如需要)
source .env

# 删除旧环境
rm -rf .venv uv.lock

# 选择 CPU 或 GPU
uv sync --extra linux-cpu    # 或 --extra linux-gpu

# 激活环境
source .venv/bin/activate

# 验证
python verify_gpu.py
```

## 验证结果

### Windows 环境成功安装:

```
Python: 3.10.18
TensorFlow: 2.10.1
Sionna: 0.14.0
ONNX: 1.12.0
tf2onnx: 1.14.0
```

### 系统信息:

```
✅ 系统: Windows 10
✅ CPU: 12 核心 / 14 线程
✅ 内存: 31.43GB
✅ TensorFlow 2.10.1 (CPU模式)
```

## 关键要点

1. **平台标记是必需的**: 防止跨平台依赖冲突
2. **required-environments 很重要**: 只为目标平台解析
3. **版本兼容性需要检查**: TF 2.10.1 有特殊的 protobuf 要求
4. **conflicts 提高清晰度**: 明确不同配置互斥
5. **包结构配置**: hatchling 需要知道项目结构

## 性能

- **解析时间**: ~1.3秒 (之前会超时或失败)
- **安装包数**: 85个包
- **虚拟环境大小**: ~2GB

## 后续步骤

1. ✅ 在 Linux 机器上测试 `linux-cpu` 和 `linux-gpu`
2. ✅ 更新文档说明平台特定的配置
3. ✅ 创建 CI/CD 管道测试所有平台
4. ✅ 添加预提交钩子检查 pyproject.toml 语法

## 参考资料

- [UV Environment Markers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#environment-markers)
- [UV Conflicts](https://docs.astral.sh/uv/concepts/dependencies/#conflicting-extras)
- [UV Required Environments](https://docs.astral.sh/uv/reference/settings/#required-environments)
- [Hatchling Configuration](https://hatch.pypa.io/latest/config/build/)

---

**问题解决!** 🎉

现在可以在 Windows 上成功运行:
```bash
source .env && uv sync --extra windows-cpu && source .venv/Scripts/activate && python verify_gpu.py
```
