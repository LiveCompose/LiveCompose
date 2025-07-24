import tensorflow as tf
import os

print("=== TensorFlow GPU 诊断 ===")
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

# 检查 CUDA 库路径
print("\n=== CUDA 环境变量 ===")
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'Not set'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))

# 详细的设备信息
print("\n=== 设备信息 ===")
print("Physical devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# 修正 GPU 配置方法
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # 使用正确的方法名
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU 内存增长设置成功")
    else:
        print("! 未检测到 GPU 设备")
except Exception as e:
    print(f"GPU 配置错误: {e}")

# 检查 TensorFlow 期望的 CUDA 版本
print("\n=== TensorFlow CUDA 构建信息 ===")
print("Expected CUDA version:", tf.sysconfig.get_build_info().get('cuda_version', 'Unknown'))
print("Expected cuDNN version:", tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown'))

# 测试 GPU 计算
print("\n=== GPU 计算测试 ===")
try:
    # 强制在 GPU 上运行
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("GPU 计算成功!")
        print("矩阵乘法结果:")
        print(c.numpy())
        print(f"计算设备: {c.device}")
except Exception as e:
    print(f"! GPU 计算失败: {e}")

# 显示 GPU 详细信息
print("\n=== GPU 详细信息 ===")
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    for i, device in enumerate(gpu_devices):
        print(f"GPU {i}: {device}")
        # 获取 GPU 详细信息
        details = tf.config.experimental.get_device_details(device)
        print(f"  设备详情: {details}")
except Exception as e:
    print(f"获取 GPU 详情失败: {e}")