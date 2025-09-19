#!/usr/bin/env python3
"""
I5R5_OHLCV模型快速测试脚本
快速验证所有模型的加载和基本预测功能
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from __init__ import *
import utils as _U
import model as _M
import dataset as _D

def quick_test_all_models():
    """快速测试所有模型"""
    print("I5R5_OHLCV模型快速测试")
    print("=" * 40)
    
    config_dir = "configs/I5R5"
    model_dir = "models/I5R5_OHLCV"
    
    # 获取所有配置文件
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
    config_files.sort()
    
    results = []
    
    for config_file in config_files:
        print(f"\n测试: {config_file}")
        
        try:
            # 加载配置
            config_path = os.path.join(config_dir, config_file)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            setting = _U.Dict2ObjParser(config).parse()
            
            # 生成模型文件名
            model_file = config_file.replace('.yml', '.tar').replace('I5R5_', 'I5R5_OHLCV_')
            model_path = os.path.join(model_dir, model_file)
            
            # 检查模型文件
            if not os.path.exists(model_path):
                print(f"  ❌ 模型文件不存在: {model_file}")
                continue
            
            # 加载模型
            model = _M.CNN5d()
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 创建测试数据
            test_dataset = _D.ImageDataSet(
                win_size=setting.DATASET.LOOKBACK_WIN,
                start_date=setting.TEST.START_DATE,
                end_date=setting.TEST.END_DATE,
                mode='test',
                label=setting.TRAIN.LABEL,
                indicators=setting.DATASET.INDICATORS,
                show_volume=setting.DATASET.SHOW_VOLUME,
                parallel_num=setting.DATASET.PARALLEL_NUM
            )
            
            # 生成少量测试数据
            test_images = test_dataset.generate_images(0.1)  # 只使用10%的数据进行快速测试
            
            if len(test_images) == 0:
                print(f"  ❌ 没有生成测试数据")
                continue
            
            # 准备测试数据
            # 模型期望输入格式: [batch_size, height, width]，模型内部会添加通道维度
            image_tensors = []
            for img in test_images[:10]:  # 只测试前10个样本
                img_tensor = torch.FloatTensor(img[0])  # img[0] 是numpy数组，形状为 [height, width]
                image_tensors.append(img_tensor)
            
            images = torch.stack(image_tensors)  # [batch_size, height, width]
            labels = torch.LongTensor([img[1] for img in test_images[:10]]) if setting.TRAIN.LABEL == 'RET5' else torch.LongTensor([img[2] for img in test_images[:10]])
            
            print(f"  🔍 张量形状: {images.shape}")
            
            # 进行预测
            with torch.no_grad():
                
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
            
            # 计算基本准确率
            accuracy = (predictions == labels).float().mean().item()
            
            print(f"  ✅ 模型加载成功")
            print(f"  📊 测试样本数: {len(test_images)}")
            print(f"  🎯 快速准确率: {accuracy:.4f}")
            print(f"  📅 训练周期: {setting.DATASET.START_DATE}-{setting.DATASET.END_DATE}")
            print(f"  🧪 测试周期: {setting.TEST.START_DATE}-{setting.TEST.END_DATE}")
            print(f"  🏷️  标签类型: {setting.TRAIN.LABEL}")
            print(f"  📈 包含成交量: {setting.DATASET.SHOW_VOLUME}")
            
            results.append({
                'config': config_file,
                'model': model_file,
                'accuracy': accuracy,
                'test_samples': len(test_images),
                'train_period': f"{setting.DATASET.START_DATE}-{setting.DATASET.END_DATE}",
                'test_period': f"{setting.TEST.START_DATE}-{setting.TEST.END_DATE}",
                'label': setting.TRAIN.LABEL,
                'show_volume': setting.DATASET.SHOW_VOLUME
            })
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            continue
    
    # 生成汇总报告
    if results:
        print(f"\n{'='*50}")
        print("快速测试汇总报告")
        print(f"{'='*50}")
        
        df = pd.DataFrame(results)
        
        print(f"成功测试模型数: {len(results)}")
        print(f"平均准确率: {df['accuracy'].mean():.4f}")
        print(f"最高准确率: {df['accuracy'].max():.4f}")
        print(f"最低准确率: {df['accuracy'].min():.4f}")
        
        print(f"\n最佳模型:")
        best_model = df.loc[df['accuracy'].idxmax()]
        print(f"  配置文件: {best_model['config']}")
        print(f"  模型文件: {best_model['model']}")
        print(f"  准确率: {best_model['accuracy']:.4f}")
        print(f"  训练周期: {best_model['train_period']}")
        print(f"  测试周期: {best_model['test_period']}")
        
        # 保存结果
        df.to_csv('quick_test_results.csv', index=False)
        print(f"\n详细结果已保存到: quick_test_results.csv")
        
        # 显示所有模型结果
        print(f"\n所有模型结果:")
        print(df[['config', 'accuracy', 'test_samples', 'train_period']].to_string(index=False))
        
    else:
        print("没有成功测试任何模型")

if __name__ == "__main__":
    quick_test_all_models()
