#!/bin/bash
# GPU 环境激活脚本
# 用途: 激活虚拟环境并配置 CUDA 库路径
# 使用: source activate_gpu.sh

# 获取脚本所在目录(项目根目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🚀 激活 Neural RX GPU 环境..."
echo ""

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

# 取消系统 CUDA 路径可能的干扰
unset CUDA_HOME

# 设置 TensorFlow 日志级别(抑制警告)
export TF_CPP_MIN_LOG_LEVEL=3  # 只显示 ERROR
export TF_ENABLE_ONEDNN_OPTS=0  # 禁用 oneDNN 警告

echo "✅ 虚拟环境: $VIRTUAL_ENV"
echo "✅ CUDA 库路径已设置"
echo "✅ TensorFlow 警告已抑制"
echo ""

# 验证 GPU
echo "🔍 验证 GPU 可用性..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'✅ 检测到 {len(gpus)} 个 GPU'); [print(f'   - {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "❌ GPU 检测失败"

echo ""
echo "📋 可用命令:"
echo "   python verify_gpu.py                                  # 完整系统验证"
echo "   python scripts/train_neural_rx.py -config_name <cfg>  # 训练模型"
echo "   python scripts/evaluate.py -config_name <cfg>         # 评估模型"
echo ""
