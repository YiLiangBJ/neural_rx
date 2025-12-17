#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# training of the neural receiver for a given configuration file
# the training loop can be found in utils.training_loop

####################################################################
# Parse args
####################################################################

import argparse
from os.path import exists

parser = argparse.ArgumentParser()
# the config defines the sys parameters
parser.add_argument("-config_name", help="config filename", type=str)
# GPU to use
parser.add_argument("-gpu", 
                    help="GPU selection: specific GPU number (0,1,2...), 'all' for all GPUs, or 'cpu' for CPU only", 
                    type=str, 
                    default="0")
# Easier debugging with breakpoints when running the code eagerly
parser.add_argument("-debug", help="Enable debug mode (disables XLA, enables eager execution)", action="store_true", default=False)
# Disable XLA compilation (faster startup, slower training)
parser.add_argument("--no-xla", help="Disable XLA compilation (useful for debugging)", action="store_true", default=False)

# Parse all arguments
args = parser.parse_args()

####################################################################
# Imports and GPU configuration
####################################################################

# Avoid warnings from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Initialize project paths (must be done before other imports)
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.project_paths import init_project_paths, get_weights_path, get_logs_path
init_project_paths()  # Switch to project root and create directories

# Configure GPU/CPU usage
gpus = tf.config.list_physical_devices('GPU')
gpu_strategy = None

if args.gpu.lower() == 'cpu':
    # Force CPU only
    tf.config.set_visible_devices([], 'GPU')
    print('🖥️  使用 CPU 训练 (所有 GPU 已禁用)')
    print('   ⚠️  警告: CPU 训练会非常慢!')
    
elif args.gpu.lower() == 'all':
    # Use all available GPUs
    if len(gpus) == 0:
        print('❌ 未检测到 GPU,切换到 CPU 模式')
        tf.config.set_visible_devices([], 'GPU')
    elif len(gpus) == 1:
        print(f'📊 检测到 1 个 GPU,自动使用')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print(f'📊 使用所有 {len(gpus)} 个 GPU 进行分布式训练')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Create multi-GPU strategy
        gpu_strategy = tf.distribute.MirroredStrategy()
        print(f'   策略: {gpu_strategy.__class__.__name__}')
        print(f'   GPU 列表: {[gpu.name for gpu in gpus]}')
        
else:
    # Use specific GPU
    try:
        gpu_id = int(args.gpu)
        if gpu_id < 0 or gpu_id >= len(gpus):
            print(f'❌ GPU {gpu_id} 不存在! 可用 GPU 数量: {len(gpus)}')
            print(f'   可用选项: 0-{len(gpus)-1}, "all", 或 "cpu"')
            sys.exit(1)
        
        # Set only the specified GPU visible
        tf.config.set_visible_devices([gpus[gpu_id]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        print(f'🎯 使用 GPU {gpu_id}: {gpus[gpu_id].name}')
        print(f'   已启用内存增长模式')
        
    except ValueError:
        print(f'❌ 无效的 GPU 参数: {args.gpu}')
        print(f'   有效选项: 0-{len(gpus)-1}, "all", 或 "cpu"')
        sys.exit(1)

print()

from utils import E2E_Model, training_loop, Parameters, load_weights

##################################################################
# Training parameters
##################################################################

# all relevant parameters are defined in the config_file
config_name = args.config_name

# initialize system parameters
sys_parameters = Parameters(config_name,
                            system='nrx',
                            training=True)
label = f'{sys_parameters.label}'
filename = get_weights_path(label)
training_logdir = get_logs_path()
training_seed = 42

# Debug mode: disable XLA and enable eager execution
if args.debug:
    tf.config.run_functions_eagerly(True)
    training_logdir = get_logs_path("debug")
    # Override XLA setting in debug mode
    sys_parameters.xla = False
    print("🐛 调试模式已激活:")
    print("   - Eager execution: 启用 (可以设置断点)")
    print("   - XLA 编译: 禁用 (无编译等待)")
    print("   - 日志目录: logs/debug/")
    print("   ⚠️  注意: 调试模式会显著降低训练速度!")
    print()

# Optional: disable XLA without full debug mode
if args.no_xla:
    sys_parameters.xla = False
    print("⚡ XLA 编译已禁用")
    print("   ✅ 优点: 无编译等待,快速启动")
    print("   ⚠️  缺点: 训练速度较慢")
    print()

#################################################################
# Start training
#################################################################

print("\n" + "=" * 70)
print("🚀 开始训练")
print("=" * 70)
print(f"📋 配置: {config_name}")
print(f"🏷️  标签: {label}")

# GPU info
if args.gpu.lower() == 'cpu':
    print(f"🖥️  计算设备: CPU")
elif args.gpu.lower() == 'all':
    print(f"🖥️  计算设备: {len(gpus)} 个 GPU (分布式训练)")
    if gpu_strategy:
        print(f"   策略: {gpu_strategy.__class__.__name__}")
else:
    print(f"🖥️  计算设备: GPU {args.gpu}")

print(f"💾 权重路径: {filename}")
print(f"📊 日志路径: {training_logdir}")
print(f"🌱 随机种子: {training_seed}")
print(f"🐛 调试模式: {'启用' if args.debug else '禁用'}")
if args.debug:
    print(f"   ⚠️  调试模式会禁用 XLA 并启用 eager execution")
print("=" * 70)
print()

# Create model (with multi-GPU strategy if applicable)
if gpu_strategy:
    print("\n🔧 在分布式策略中创建模型...")
    with gpu_strategy.scope():
        sys_training = E2E_Model(sys_parameters, training=True)
        sys_training(1, 1.)  # run once to init weights
    print("✅ 分布式模型创建完成")
else:
    sys_training = E2E_Model(sys_parameters, training=True)
    sys_training(1, 1.)  # run once to init weights in TensorFlow

sys_training.summary()

# load weights if the exists already
if exists(filename):
    print("\n💡 检测到已有权重 - 加载中...")
    load_weights(sys_training, filename)
    print("✅ 权重加载完成")
else:
    print("\n🆕 从头开始训练 (未找到已有权重)")

print()
print("⚙️  训练参数:")
print(f"   📚 Epochs: {sys_parameters.training_schedule['epochs']}")
print(f"   📦 Batch size: {sys_parameters.training_schedule['batch_size']}")
print(f"   👥 用户数范围: {sys_parameters.min_num_tx} - {sys_parameters.max_num_tx}")
print(f"   📡 MCS 索引: {sys_parameters.mcs_index}")
print(f"   📈 评估 EbNo: {sys_parameters.eval_ebno_db_arr} dB")
print(f"   ⚡ XLA 加速: {sys_parameters.xla}")
print()

if hasattr(sys_parameters, 'mcs_training_snr_db_offset'):
    mcs_training_snr_db_offset = sys_parameters.mcs_training_snr_db_offset
else:
    mcs_training_snr_db_offset = None

if hasattr(sys_parameters, 'mcs_training_probs'):
    mcs_training_probs = sys_parameters.mcs_training_probs
else:
    mcs_training_probs = None

print("🎬 启动训练循环...")
print("=" * 70)
print()

# run the training / weights are automatically saved
# UEs' MCSs will be drawn randomly
training_loop(sys_training,
              label=label,
              filename=filename,
              training_logdir=training_logdir,
              training_seed=training_seed,
              training_schedule=sys_parameters.training_schedule,
              eval_ebno_db_arr=sys_parameters.eval_ebno_db_arr,
              min_num_tx=sys_parameters.min_num_tx,
              max_num_tx=sys_parameters.max_num_tx,
              sys_parameters=sys_parameters,
              mcs_arr_training_idx=list(range(len(sys_parameters.mcs_index))), # train with all supported MCSs
              mcs_training_snr_db_offset=mcs_training_snr_db_offset,
              mcs_training_probs=mcs_training_probs,
              xla=sys_parameters.xla)

print()
print("=" * 70)
print("✅ 训练完成!")
print(f"💾 最终权重: {filename}")
print(f"📊 TensorBoard: tensorboard --logdir {training_logdir}")
print("=" * 70)
