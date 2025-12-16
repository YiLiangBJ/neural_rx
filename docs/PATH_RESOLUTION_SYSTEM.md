# 🔧 统一路径解析系统说明

## 问题背景

原始代码设计为从 `scripts/` 目录内运行,所有路径都使用相对路径 `../`:
- `../config/xxx.cfg`
- `../weights/xxx_weights`
- `../results/xxx_results`
- `../onnx_models/xxx.onnx`

这导致:
- ❌ 必须从 `scripts/` 目录运行
- ❌ 从项目根目录运行会找不到文件
- ❌ 路径字符串分散在多个文件中
- ❌ 难以维护和调试

---

## 解决方案

创建了 **`utils/project_paths.py`** 统一路径管理模块。

### 核心功能

1. **自动查找项目根目录**
   ```python
   # 通过查找 pyproject.toml 定位项目根目录
   PROJECT_ROOT = get_project_root()
   ```

2. **预定义所有资源目录**
   ```python
   CONFIG_DIR = PROJECT_ROOT / 'config'
   WEIGHTS_DIR = PROJECT_ROOT / 'weights'
   RESULTS_DIR = PROJECT_ROOT / 'results'
   LOGS_DIR = PROJECT_ROOT / 'logs'
   ONNX_DIR = PROJECT_ROOT / 'onnx_models'
   ```

3. **提供路径辅助函数**
   ```python
   get_config_path(config_name)   # 配置文件路径
   get_weights_path(label)         # 权重文件路径
   get_results_path(label)         # 结果文件路径
   get_logs_path(label='')         # 日志路径
   get_onnx_path(label, ext)       # ONNX 模型路径
   ```

4. **自动创建必需目录**
   ```python
   init_project_paths()  # 创建所有目录并切换到项目根
   ```

---

## 使用方法

### 在脚本中使用

```python
# 脚本开头
from utils.project_paths import init_project_paths, get_weights_path, get_logs_path
init_project_paths()  # 必须先调用!

# 然后使用路径函数
weights_file = get_weights_path('nrx_large')  # 返回: PROJECT_ROOT/weights/nrx_large_weights
logs_dir = get_logs_path()  # 返回: PROJECT_ROOT/logs
```

### 在 utils 模块中使用

```python
# 直接导入需要的路径或函数
from utils.project_paths import WEIGHTS_DIR, get_config_path

# 使用
config_file = get_config_path('nrx_large')  # 自动添加 .cfg
weights_file = WEIGHTS_DIR / 'model_weights'
```

---

## 已更新的文件

### 脚本 (scripts/)

1. **train_neural_rx.py**
   ```python
   # 之前
   filename = '../weights/' + label + '_weights'
   training_logdir = '../logs'
   
   # 现在
   filename = get_weights_path(label)
   training_logdir = get_logs_path()
   ```

2. **evaluate.py**
   ```python
   # 之前
   results_filename = f"../results/{sys_parameters.label}_results"
   filename = f'../weights/{sys_parameters.label}_weights'
   
   # 现在
   results_filename = get_results_path(sys_parameters.label)
   filename = get_weights_path(sys_parameters.label)
   ```

3. **export_onnx.py**
   ```python
   # 之前
   load_weights(neural_rx, f"../weights/{sys_parameters.label}_weights")
   neural_rx.save(f"../onnx_models/{sys_parameters.label}_tf")
   onnx.save(onnx_model, f"../onnx_models/{sys_parameters.label}.onnx")
   
   # 现在
   load_weights(neural_rx, get_weights_path(sys_parameters.label))
   neural_rx.save(get_onnx_path(sys_parameters.label, "_tf"))
   onnx.save(onnx_model, get_onnx_path(sys_parameters.label, ".onnx"))
   ```

4. **compute_cov_mat.py**
   ```python
   # 之前
   np.save(f'../weights/{parameters.label}_freq_cov_mat', freq_cov_mat)
   
   # 现在
   np.save(str(WEIGHTS_DIR / f'{parameters.label}_freq_cov_mat'), freq_cov_mat)
   ```

### Utils 模块 (utils/)

1. **parameters.py**
   ```python
   # 之前
   fn = f'../config/{config_name}'
   
   # 现在
   from utils.project_paths import get_config_path
   fn = get_config_path(config_name)  # 自动添加 .cfg 扩展名
   ```

2. **utils.py**
   ```python
   # 之前
   filename = f"../results/{sys_parameters.label}_results"
   filename = f'../weights/{sys_parameters.label}_weights'
   
   # 现在
   from utils.project_paths import get_results_path, get_weights_path
   filename = get_results_path(sys_parameters.label)
   filename = get_weights_path(sys_parameters.label)
   ```

---

## 优势

### ✅ 灵活性
```bash
# 现在可以从任何地方运行!
cd ~/neural_rx
python scripts/train_neural_rx.py -config_name nrx_large

# 或
cd ~/neural_rx/scripts
python train_neural_rx.py -config_name nrx_large

# 都可以正常工作!
```

### ✅ 可维护性
```python
# 所有路径定义在一个地方
# 修改目录结构只需更新 project_paths.py
```

### ✅ 可靠性
```python
# 自动创建目录,避免 FileNotFoundError
init_project_paths()  # weights/, results/, logs/ 自动创建
```

### ✅ 清晰性
```python
# 之前
filename = '../weights/nrx_large_weights'  # 什么路径?从哪里?

# 现在
filename = get_weights_path('nrx_large')  # 清晰明确!
```

---

## 迁移指南

如果你有自定义脚本,按以下步骤迁移:

### 步骤 1: 导入路径模块

```python
# 在脚本开头添加
from utils.project_paths import init_project_paths, get_weights_path, get_results_path
init_project_paths()
```

### 步骤 2: 替换路径字符串

| 旧代码 | 新代码 |
|-------|--------|
| `'../config/xxx.cfg'` | `get_config_path('xxx')` |
| `'../weights/xxx_weights'` | `get_weights_path('xxx')` |
| `'../results/xxx_results'` | `get_results_path('xxx')` |
| `'../logs'` | `get_logs_path()` |
| `'../onnx_models/xxx.onnx'` | `get_onnx_path('xxx', '.onnx')` |

### 步骤 3: 测试

```bash
# 从不同目录测试
cd ~/neural_rx
python your_script.py

cd ~/neural_rx/scripts
python your_script.py
```

---

## API 参考

### `init_project_paths()`
初始化项目路径系统(必须在脚本开头调用)
- 切换到项目根目录
- 创建所有必需目录
- 返回项目根路径

### `get_config_path(config_name)`
获取配置文件路径
- 参数: `config_name` - 配置名(自动添加 `.cfg`)
- 返回: 完整路径字符串

### `get_weights_path(label)`
获取权重文件路径
- 参数: `label` - 模型标签
- 返回: `PROJECT_ROOT/weights/{label}_weights`

### `get_results_path(label)`
获取结果文件路径
- 参数: `label` - 结果标签
- 返回: `PROJECT_ROOT/results/{label}_results`

### `get_logs_path(label='')`
获取日志目录路径
- 参数: `label` - 可选子目录
- 返回: `PROJECT_ROOT/logs` 或 `PROJECT_ROOT/logs/{label}`

### `get_onnx_path(label, extension='')`
获取 ONNX 模型路径
- 参数: `label` - 模型标签, `extension` - 文件扩展名
- 返回: `PROJECT_ROOT/onnx_models/{label}{extension}`

### 常量

- `PROJECT_ROOT` - 项目根目录 (Path 对象)
- `CONFIG_DIR` - 配置目录
- `WEIGHTS_DIR` - 权重目录
- `RESULTS_DIR` - 结果目录
- `LOGS_DIR` - 日志目录
- `ONNX_DIR` - ONNX 模型目录

---

## 测试

```bash
# 测试路径模块
cd ~/neural_rx
python -c "from utils.project_paths import *; init_project_paths(); print('PROJECT_ROOT:', PROJECT_ROOT); print('Config:', get_config_path('nrx_large')); print('Weights:', get_weights_path('test'))"

# 预期输出:
# PROJECT_ROOT: /home/xxx/neural_rx
# Config: /home/xxx/neural_rx/config/nrx_large.cfg
# Weights: /home/xxx/neural_rx/weights/test_weights
```

---

## 故障排除

### Q: 脚本找不到 `utils.project_paths`

**A**: 确保在导入前添加项目根到 `sys.path`:
```python
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.project_paths import init_project_paths
```

### Q: `init_project_paths()` 失败

**A**: 确保项目根目录有 `pyproject.toml` 文件。

### Q: 路径不正确

**A**: 检查是否调用了 `init_project_paths()`:
```python
# 错误
from utils.project_paths import get_weights_path
path = get_weights_path('test')  # 可能不正确

# 正确
from utils.project_paths import init_project_paths, get_weights_path
init_project_paths()  # 先初始化!
path = get_weights_path('test')  # 现在正确
```

---

## 总结

**统一路径解析系统**解决了所有路径相关的问题:

- ✅ 从任何目录运行脚本
- ✅ 自动创建必需目录
- ✅ 集中管理所有路径
- ✅ 清晰易维护
- ✅ 类型安全 (使用 Path 对象)

**所有脚本现在都可以正常工作!** 🎉
