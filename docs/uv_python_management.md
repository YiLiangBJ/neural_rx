# UV Python 版本管理说明

## 🎯 UV 如何管理 Python 版本

UV 是一个现代化的 Python 包管理器,具有**自动 Python 版本管理**功能。

### 工作原理

1. **读取配置**
   - `.python-version` 文件: `3.10`
   - `pyproject.toml`: `requires-python = ">=3.10,<3.11"`

2. **查找 Python**
   - 在系统中查找 Python 3.10
   - 检查 PATH 环境变量
   - 检查常见安装位置

3. **自动下载(如果需要)**
   - 如果本地没有 Python 3.10
   - UV 会从官方源自动下载
   - 安装到 UV 的缓存目录
   - 自动配置虚拟环境使用

### 本项目的配置

```
neural_rx/
├── .python-version          # 指定 Python 3.10
├── pyproject.toml          # requires-python = ">=3.10,<3.11"
└── .env                    # 代理配置(用于下载 Python)
```

### 为什么需要代理配置?

UV 下载 Python 需要访问:
- `https://github.com/astral-sh/python-build-standalone`
- 或其他 Python 分发源

Intel 内网用户需要通过代理访问,所以:

```bash
# 1. 必须先加载代理
source .env

# 2. 然后 UV 才能下载 Python
uv sync --extra windows-cpu
```

### UV 的三种 Python 策略

UV 支持三种 Python 查找策略(通过 `UV_PYTHON_PREFERENCE` 环境变量):

| 策略 | 行为 | 本项目使用 |
|------|------|-----------|
| `managed` (默认) | 优先使用 UV 管理的 Python,没有则下载 | ✅ 是 |
| `system` | 优先系统 Python,没有再下载 | ❌ 否 |
| `only-system` | 只用系统 Python,没有就报错 | ❌ 否 |

**我们使用默认策略 `managed`**,让 UV 自动处理一切!

### 实际使用流程

#### 场景 1: 系统已有 Python 3.10

```bash
source .env
uv sync --extra windows-cpu

# UV 输出:
# Using Python 3.10.11 at C:\Users\...\Python310\python.exe
# ✅ 直接使用系统 Python
```

#### 场景 2: 系统没有 Python 3.10

```bash
source .env
uv sync --extra windows-cpu

# UV 输出:
# Downloading Python 3.10.15 from github.com/astral-sh/python-build-standalone
# Installing to C:\Users\...\AppData\Local\uv\python\...
# Using Python 3.10.15 at ...
# ✅ 自动下载并使用
```

#### 场景 3: 没有代理配置

```bash
uv sync --extra windows-cpu

# ❌ 错误:
# error: Request failed after 3 retries
# Caused by: Failed to download https://github.com/...
# Caused by: tcp connect error (timeout)

# 解决: source .env
```

### 常见问题

#### Q1: 我需要手动安装 Python 3.10 吗?

**A**: 不需要! UV 会自动下载。但你必须先 `source .env` 加载代理。

#### Q2: UV 下载的 Python 在哪里?

**A**: 在 UV 的缓存目录:
- Windows: `C:\Users\<用户>\AppData\Local\uv\python\`
- Linux: `~/.local/share/uv/python/`

#### Q3: 我可以使用系统的 Python 吗?

**A**: 可以!如果系统有 Python 3.10,UV 会优先使用。

#### Q4: 如何强制 UV 下载新的 Python?

**A**: 
```bash
uv python install 3.10  # 明确安装 Python 3.10
uv sync --extra windows-cpu
```

#### Q5: 多个项目都用 UV,会重复下载 Python 吗?

**A**: 不会!UV 会缓存 Python,所有项目共享。

### 优势

✅ **无需手动安装**: UV 自动处理 Python 版本
✅ **版本隔离**: 不同项目可以用不同 Python 版本
✅ **可重现**: 团队成员都用相同的 Python 版本
✅ **节省空间**: 多个项目共享缓存的 Python
✅ **跨平台**: Windows/Linux/macOS 行为一致

### 对比传统方式

| 方面 | 传统方式 | UV 方式 |
|------|---------|---------|
| 安装 Python | 手动从官网下载安装 | UV 自动下载 |
| 版本管理 | 手动切换 PATH | UV 自动选择 |
| 虚拟环境 | `python -m venv` | `uv sync` |
| 依赖安装 | `pip install` | `uv sync` |
| 锁定文件 | 手动维护 | 自动 `uv.lock` |

### 总结

**本项目配置的关键点**:

1. ✅ `.python-version` + `pyproject.toml` 指定 Python 3.10
2. ✅ `.env` 配置代理(Intel 内网必需)
3. ✅ UV 默认策略 `managed` - 自动下载 Python
4. ❌ 不需要 `UV_PYTHON_PREFERENCE=only-system`
5. ❌ 不需要 `UV_PYTHON=/path/to/python`
6. ❌ 不需要手动安装 Python 3.10

**一条命令搞定**:
```bash
source .env && uv sync --extra windows-cpu
```

UV 会自动处理其余所有事情! 🚀

### 参考资料

- [UV 官方文档 - Python 版本](https://github.com/astral-sh/uv#python-versions)
- [UV Python 安装](https://docs.astral.sh/uv/guides/install-python/)
- [UV 配置选项](https://docs.astral.sh/uv/configuration/)
