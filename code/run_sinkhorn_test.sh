#!/bin/bash

# 测试 Self-Attention GAT + Sinkhorn 拟合 Hungarian 结果的运行脚本
# 使用方法: bash run_sinkhorn_test.sh

echo "=================================================="
echo "Self-Attention GAT + Sinkhorn Hungarian 拟合测试"
echo "=================================================="

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda命令"
    exit 1
fi

# 激活conda环境
echo "🔄 激活conda环境: MAPD_RL"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MAPD_RL

if [ $? -ne 0 ]; then
    echo "❌ 错误: 无法激活conda环境 MAPD_RL"
    exit 1
fi

echo "✅ 成功激活conda环境: MAPD_RL"

# 检查当前目录
CURRENT_DIR=$(pwd)
TARGET_DIR="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code"

if [ "$CURRENT_DIR" != "$TARGET_DIR" ]; then
    echo "🔄 切换到目标目录: $TARGET_DIR"
    cd "$TARGET_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 无法切换到目录 $TARGET_DIR"
        exit 1
    fi
fi

echo "✅ 当前工作目录: $(pwd)"

# 检查测试脚本是否存在
if [ ! -f "test_sinkhorn_hungarian_fit.py" ]; then
    echo "❌ 错误: 测试脚本 test_sinkhorn_hungarian_fit.py 不存在"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="$TARGET_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # 使用第一块GPU

echo "🚀 开始运行测试..."
echo "设备: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python路径: $PYTHONPATH"
echo ""

# 运行测试
python test_sinkhorn_hungarian_fit.py

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试脚本执行完成"
    
    # 检查结果图片是否生成
    if [ -f "sinkhorn_hungarian_fit_results.png" ]; then
        echo "📊 结果图片已生成: sinkhorn_hungarian_fit_results.png"
    fi
else
    echo ""
    echo "❌ 测试脚本执行失败"
    exit 1
fi

echo ""
echo "🎉 所有任务完成！" 