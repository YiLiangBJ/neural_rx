#!/bin/bash
# 激活虚拟环境并设置 CUDA 库路径

# 获取脚本所在目录(项目根目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活虚拟环境
source "$SCRIPT_DIR/.venv/bin/activate"

# 设置 CUDA 库路径
CUDA_LIB_PATHS=(
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/cublas/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/cufft/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/curand/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/cusolver/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib"
    "$SCRIPT_DIR/.venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
)

# 构建 LD_LIBRARY_PATH
CUDA_LIB_PATH=$(IFS=:; echo "${CUDA_LIB_PATHS[*]}")
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"

# 同时取消系统 CUDA 路径可能的干扰
unset CUDA_HOME

echo "✅ 虚拟环境已激活"
echo "✅ CUDA 库路径已设置"
echo ""
echo "验证 GPU:"
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('检测到 GPU 数量:', len(gpus)); [print(f'  - {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "❌ GPU 检测失败"
echo ""
echo "现在可以运行训练了:"
echo "  python scripts/train_neural_rx.py -config_name nrx_large -gpu 0"
