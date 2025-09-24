#!/bin/bash

# CNN-for-Trading 论文标准实验脚本
# 按照论文要求运行完整的实验流程

echo "=========================================="
echo "CNN-for-Trading 论文标准实验"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查数据目录
if [ ! -d "data" ]; then
    echo "错误: 未找到data目录，请将CSV数据文件放入data/目录"
    exit 1
fi

# 检查配置文件
if [ ! -d "configs" ]; then
    echo "错误: 未找到configs目录"
    exit 1
fi

# 创建结果目录
mkdir -p results
mkdir -p models_2
mkdir -p logs_2

echo "开始运行论文标准实验..."

# 实验1: 5天预测模型
echo ""
echo "=========================================="
echo "实验1: 5天预测模型 (I5R5)"
echo "=========================================="
python main.py configs/I5R5/I5R5_93-00_train.yml \
    --multiple_training \
    --portfolio_evaluation \
    --benchmark_comparison

if [ $? -eq 0 ]; then
    echo "✅ 5天预测模型实验完成"
else
    echo "❌ 5天预测模型实验失败"
fi

# 实验2: 20天预测模型
echo ""
echo "=========================================="
echo "实验2: 20天预测模型 (I20R20)"
echo "=========================================="
python main.py configs/I20R20/I20R20_93-00_train.yml \
    --multiple_training \
    --portfolio_evaluation \
    --benchmark_comparison

if [ $? -eq 0 ]; then
    echo "✅ 20天预测模型实验完成"
else
    echo "❌ 20天预测模型实验失败"
fi

# 实验3: 60天预测模型
echo ""
echo "=========================================="
echo "实验3: 60天预测模型 (I60R60)"
echo "=========================================="
python main.py configs/I60R60/I60R60_93-00_train.yml \
    --multiple_training \
    --portfolio_evaluation \
    --benchmark_comparison

if [ $? -eq 0 ]; then
    echo "✅ 60天预测模型实验完成"
else
    echo "❌ 60天预测模型实验失败"
fi

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="

# 显示结果摘要
echo ""
echo "📊 实验结果摘要:"
echo "模型文件保存在: models_2/"
echo "训练日志保存在: logs_2/"
echo "投资组合评估结果保存在: results/"

echo ""
echo "📈 主要输出文件:"
ls -la results/*.csv 2>/dev/null || echo "  暂无CSV结果文件"
ls -la results/*.png 2>/dev/null || echo "  暂无PNG图表文件"

echo ""
echo "🎯 实验完成！请查看results/目录中的详细结果。"

