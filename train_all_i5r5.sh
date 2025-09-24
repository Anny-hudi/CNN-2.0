#!/bin/bash

# I5R5模型批量训练脚本
# 所有模型结果将保存到 I5R5_OHLCV/ 目录

cd /root/pythonprojects/new_CNN/CNN-for-Trading

echo "开始训练所有I5R5模型..."

# 训练I5R5_09-11
echo "训练 I5R5_09-11 模型..."
python main.py configs/I5R5/I5R5_09-11.yml

# 训练I5R5_10-12
echo "训练 I5R5_10-12 模型..."
python main.py configs/I5R5/I5R5_10-12.yml

# 训练I5R5_11-13
echo "训练 I5R5_11-13 模型..."
python main.py configs/I5R5/I5R5_11-13.yml

# 训练I5R5_12-14
echo "训练 I5R5_12-14 模型..."
python main.py configs/I5R5/I5R5_12-14.yml

# 训练I5R5_13-15
echo "训练 I5R5_13-15 模型..."
python main.py configs/I5R5/I5R5_13-15.yml

# 训练I5R5_14-16
echo "训练 I5R5_14-16 模型..."
python main.py configs/I5R5/I5R5_14-16.yml

# 训练I5R5_15-17
echo "训练 I5R5_15-17 模型..."
python main.py configs/I5R5/I5R5_15-17.yml

# 训练I5R5_16-18
echo "训练 I5R5_16-18 模型..."
python main.py configs/I5R5/I5R5_16-18.yml

# 训练I5R5_17-19
echo "训练 I5R5_17-19 模型..."
python main.py configs/I5R5/I5R5_17-19.yml

# 训练I5R5_18-20
echo "训练 I5R5_18-20 模型..."
python main.py configs/I5R5/I5R5_18-20.yml

echo "所有I5R5模型训练完成！"
echo "模型文件保存在: models/I5R5_OHLCV/"
echo "训练日志保存在: logs/I5R5_OHLCV/"





