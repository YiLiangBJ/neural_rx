"""
项目路径配置模块
提供统一的路径解析,无论从哪个目录运行都能正确找到文件
"""
import os
from pathlib import Path

# 获取项目根目录(包含 pyproject.toml 的目录)
def get_project_root():
    """返回项目根目录的绝对路径"""
    # 从当前文件位置向上查找 pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    # 如果找不到,假设当前就是根目录
    return Path.cwd()

# 项目根目录
PROJECT_ROOT = get_project_root()

# 常用目录路径
CONFIG_DIR = PROJECT_ROOT / 'config'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'
ONNX_DIR = PROJECT_ROOT / 'onnx_models'

# 确保目录存在
def ensure_dirs():
    """创建所有必需的目录"""
    for dir_path in [WEIGHTS_DIR, RESULTS_DIR, LOGS_DIR, ONNX_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# 路径辅助函数
def get_config_path(config_name):
    """获取配置文件路径,自动添加 .cfg 扩展名"""
    if not config_name.endswith('.cfg'):
        config_name += '.cfg'
    return str(CONFIG_DIR / config_name)

def get_weights_path(label):
    """获取权重文件路径"""
    return str(WEIGHTS_DIR / f'{label}_weights')

def get_results_path(label):
    """获取结果文件路径"""
    return str(RESULTS_DIR / f'{label}_results')

def get_logs_path(label=''):
    """获取日志路径"""
    if label:
        return str(LOGS_DIR / label)
    return str(LOGS_DIR)

def get_onnx_path(label, extension=''):
    """获取 ONNX 模型路径"""
    if extension:
        return str(ONNX_DIR / f'{label}{extension}')
    return str(ONNX_DIR / label)

# 初始化:创建目录并切换到项目根目录
def init_project_paths():
    """初始化项目路径(在脚本开始时调用)"""
    os.chdir(PROJECT_ROOT)
    ensure_dirs()
    return PROJECT_ROOT

if __name__ == '__main__':
    # 测试
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"配置目录: {CONFIG_DIR}")
    print(f"权重目录: {WEIGHTS_DIR}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print(f"ONNX 目录: {ONNX_DIR}")
