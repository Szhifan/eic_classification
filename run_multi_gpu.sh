#!/bin/bash

# 多GPU训练启动脚本
# 使用方法: ./run_multi_gpu.sh [num_gpus]
# 例如: ./run_multi_gpu.sh 2  # 使用2个GPU

# 设置GPU数量，默认为可用GPU数量
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "Starting multi-GPU training with $NUM_GPUS GPUs..."

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is installed."
fi

# 显示GPU信息
echo "Available GPUs:"
nvidia-smi -L

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export NCCL_DEBUG=INFO  # 可选：显示NCCL调试信息

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# 启动训练
if [ $NUM_GPUS -eq 1 ]; then
    echo "Starting single GPU training..."
    python train_multi_gpu.py
else
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    # 使用torchrun (推荐，适用于PyTorch 1.9+)
    torchrun --nproc_per_node=$NUM_GPUS train_multi_gpu.py
    
    # 如果torchrun不可用，使用以下命令（适用于旧版本PyTorch）
    # python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_multi_gpu.py
fi

echo "Training completed!"
