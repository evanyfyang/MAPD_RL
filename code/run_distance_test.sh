#!/bin/bash

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MAPD_RL

# 切换到代码目录
cd "$(dirname "$0")"

# 检查环境
echo "当前工作目录: $(pwd)"
echo "Python路径: $(which python)"
echo "CUDA可用性:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "开始运行距离学习测试..."
echo "=================================="

# 设置CUDA设备（如果有多个GPU）
export CUDA_VISIBLE_DEVICES=0

echo "运行优化后的距离学习测试..."
echo "主要改进："
echo "- 学习率从 1e-3 降低到 5e-5"
echo "- 增加模型非线性层数和深度"
echo "- 训练样本从 300 增加到 2000+"
echo "- 添加验证集和早停机制"
echo "- 增强模型架构（更大隐藏维度）"
echo ""

# 运行优化后的测试
python test_distance_fixed.py

echo "=================================="
echo "测试完成!"
echo ""
echo "如果结果仍不理想，可以尝试："
echo "1. 进一步降低学习率到 1e-5"
echo "2. 增加更多训练轮数"
echo "3. 调整网络深度和宽度"
echo "4. 使用不同的激活函数或优化器" 