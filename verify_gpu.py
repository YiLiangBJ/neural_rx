#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU, CUDA 和 cuDNN 验证脚本
检查 TensorFlow 和 PyTorch 的 GPU 支持情况
"""

import sys
import os

# Windows 编码修复
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def verify_tensorflow_gpu():
    """验证 TensorFlow GPU 配置"""
    print("=" * 60)
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
    print("\n🔍 GPU/CUDA/cuDNN 环境验证工具\n")
    
    tf_has_gpu = verify_tensorflow_gpu()
    pt_has_gpu = verify_pytorch_gpu()
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 60)
    
    # 返回状态码
    if tf_has_gpu or pt_has_gpu:
        print("\n✅ 至少一个框架支持 GPU")
        return 0
    else:
        print("\n⚠️  所有框架都使用 CPU 模式")
        return 1


if __name__ == "__main__":
    sys.exit(main())
