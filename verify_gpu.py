#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统能力验证脚本
检查 CPU、内存、GPU、CUDA 和 cuDNN 等硬件和软件环境
"""

import sys
import os
import platform
import psutil

# Windows 编码修复
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def get_size(bytes, suffix="B"):
    """将字节转换为人类可读的格式"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def verify_system_info():
    """验证系统基本信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    
    uname = platform.uname()
    print(f"系统: {uname.system}")
    print(f"节点名称: {uname.node}")
    print(f"发行版本: {uname.release}")
    print(f"版本: {uname.version}")
    print(f"机器类型: {uname.machine}")
    print(f"处理器: {uname.processor if uname.processor else platform.processor()}")
    
    # Python 信息
    print(f"\nPython 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")


def verify_cpu_info():
    """验证 CPU 信息"""
    print("\n" + "=" * 60)
    print("CPU 信息")
    print("=" * 60)
    
    # CPU 核心数
    print(f"物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"逻辑核心数 (含超线程): {psutil.cpu_count(logical=True)}")
    
    # CPU 频率
    try:
        cpufreq = psutil.cpu_freq()
        if cpufreq:
            print(f"最大频率: {cpufreq.max:.2f} MHz")
            print(f"最小频率: {cpufreq.min:.2f} MHz")
            print(f"当前频率: {cpufreq.current:.2f} MHz")
    except Exception:
        print("无法获取 CPU 频率信息")
    
    # CPU 使用率
    print(f"\nCPU 总体使用率: {psutil.cpu_percent(interval=1)}%")
    
    # 每个核心的使用率
    print("各核心使用率:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"  核心 {i}: {percentage}%")


def verify_memory_info():
    """验证内存信息"""
    print("\n" + "=" * 60)
    print("内存信息")
    print("=" * 60)
    
    # RAM 信息
    svmem = psutil.virtual_memory()
    print(f"总内存: {get_size(svmem.total)}")
    print(f"可用内存: {get_size(svmem.available)}")
    print(f"已用内存: {get_size(svmem.used)} ({svmem.percent}%)")
    
    # SWAP 信息
    swap = psutil.swap_memory()
    print(f"\nSWAP 总量: {get_size(swap.total)}")
    print(f"SWAP 可用: {get_size(swap.free)}")
    print(f"SWAP 已用: {get_size(swap.used)} ({swap.percent}%)")


def verify_disk_info():
    """验证磁盘信息"""
    print("\n" + "=" * 60)
    print("磁盘信息")
    print("=" * 60)
    
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"\n设备: {partition.device}")
        print(f"  挂载点: {partition.mountpoint}")
        print(f"  文件系统类型: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            print(f"  总空间: {get_size(partition_usage.total)}")
            print(f"  已用: {get_size(partition_usage.used)}")
            print(f"  可用: {get_size(partition_usage.free)}")
            print(f"  使用率: {partition_usage.percent}%")
        except PermissionError:
            print("  无权限访问此分区")


def verify_tensorflow_gpu():
    """验证 TensorFlow GPU 配置"""
    print("\n" + "=" * 60)
    print("检查 TensorFlow GPU 支持")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 版本: {tf.__version__}")
        
        # 检查 GPU 是否可用
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\n{'✅' if len(gpus) > 0 else '❌'} GPU 可用: {len(gpus) > 0}")
        
        if gpus:
            print(f"   检测到 {len(gpus)} 块 GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   - GPU {i}: {gpu.name}")
            
            # 打印 CUDA 和 cuDNN 版本
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get('cuda_version', 'N/A')
            cudnn_version = build_info.get('cudnn_version', 'N/A')
            
            print(f"\n   CUDA 版本: {cuda_version}")
            print(f"   cuDNN 版本: {cudnn_version}")
            
            # 测试简单操作
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print(f"\n   ✅ GPU 计算测试成功")
            except Exception as e:
                print(f"\n   ❌ GPU 计算测试失败: {e}")
        else:
            print("   未检测到 GPU,使用 CPU 模式")
            
    except ImportError:
        print("❌ TensorFlow 未安装")
        return False
    except Exception as e:
        print(f"❌ 检查 TensorFlow 时出错: {e}")
        return False
    
    return len(gpus) > 0 if gpus else False


def verify_pytorch_gpu():
    """验证 PyTorch GPU 配置"""
    print("\n" + "=" * 60)
    print("检查 PyTorch GPU 支持")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        
        # 检查 GPU 是否可用
        cuda_available = torch.cuda.is_available()
        print(f"\n{'✅' if cuda_available else '❌'} GPU 可用: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"   检测到 {device_count} 块 GPU:")
            
            for i in range(device_count):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # 打印 CUDA 版本
            cuda_version = torch.version.cuda
            print(f"\n   CUDA 版本: {cuda_version}")
            
            # 打印 cuDNN 版本
            if torch.backends.cudnn.enabled:
                cudnn_version = torch.backends.cudnn.version()
                print(f"   cuDNN 版本: {cudnn_version}")
                print(f"   cuDNN 已启用: {torch.backends.cudnn.enabled}")
            else:
                print(f"   cuDNN 未启用")
            
            # 测试简单操作
            try:
                x = torch.rand(3, 3).cuda()
                y = torch.rand(3, 3).cuda()
                z = torch.matmul(x, y)
                print(f"\n   ✅ GPU 计算测试成功")
            except Exception as e:
                print(f"\n   ❌ GPU 计算测试失败: {e}")
        else:
            print("   未检测到 GPU,使用 CPU 模式")
            
    except ImportError:
        print("⚠️  PyTorch 未安装 (本项目不需要)")
        return None
    except Exception as e:
        print(f"❌ 检查 PyTorch 时出错: {e}")
        return False
    
    return cuda_available


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🔍 系统能力验证工具")
    print("=" * 60 + "\n")
    
    # 系统信息
    verify_system_info()
    
    # CPU 信息
    verify_cpu_info()
    
    # 内存信息
    verify_memory_info()
    
    # 磁盘信息
    verify_disk_info()
    
    # TensorFlow GPU
    tf_has_gpu = verify_tensorflow_gpu()
    
    # PyTorch GPU
    pt_has_gpu = verify_pytorch_gpu()
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    # 系统摘要
    uname = platform.uname()
    print(f"✅ 系统: {uname.system} {uname.release}")
    print(f"✅ CPU: {psutil.cpu_count(logical=False)} 核心 / {psutil.cpu_count(logical=True)} 线程")
    
    svmem = psutil.virtual_memory()
    print(f"✅ 内存: {get_size(svmem.total)} ({get_size(svmem.available)} 可用)")
    
    # AI 框架支持
    if tf_has_gpu:
        print("✅ TensorFlow GPU 支持正常")
    else:
        print("⚠️  TensorFlow 未检测到 GPU (将使用 CPU)")
    
    if pt_has_gpu is not None:
        if pt_has_gpu:
            print("✅ PyTorch GPU 支持正常")
        else:
            print("⚠️  PyTorch 未检测到 GPU (将使用 CPU)")
    else:
        print("ℹ️  PyTorch 未安装")
    
    # 性能评估
    print("\n" + "=" * 60)
    print("性能评估")
    print("=" * 60)
    
    cpu_cores = psutil.cpu_count(logical=False)
    total_mem_gb = svmem.total / (1024**3)
    
    if tf_has_gpu or (pt_has_gpu if pt_has_gpu is not None else False):
        print("🚀 推荐用途: 深度学习训练和推理 (GPU 加速)")
    elif cpu_cores >= 8 and total_mem_gb >= 16:
        print("💻 推荐用途: 深度学习训练和推理 (CPU 模式)")
    elif cpu_cores >= 4 and total_mem_gb >= 8:
        print("📊 推荐用途: 小规模训练、推理和实验")
    else:
        print("⚠️  硬件配置较低,建议仅用于代码开发和调试")
    
    print("\n" + "=" * 60)
    
    # 返回状态码
    return 0


if __name__ == "__main__":
    sys.exit(main())
