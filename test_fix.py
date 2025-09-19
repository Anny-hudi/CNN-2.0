#!/usr/bin/env python3
"""
验证张量维度修复
"""

import os
import torch
import yaml
from __init__ import *
import utils as _U
import model as _M
import dataset as _D

def test_single_model():
    """测试单个模型验证修复"""
    print("验证张量维度修复")
    print("=" * 30)
    
    # 使用第一个配置文件
    config_file = "configs/I5R5/I5R5_17-19.yml"
    model_file = "models/I5R5_OHLCV/I5R5_OHLCV_17-19.tar"
    
    try:
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        setting = _U.Dict2ObjParser(config).parse()
        
        print(f"配置文件: {config_file}")
        print(f"模型文件: {model_file}")
        
        # 检查模型文件
        if not os.path.exists(model_file):
            print(f"❌ 模型文件不存在: {model_file}")
            return
        
        # 加载模型
        model = _M.CNN5d()
        checkpoint = torch.load(model_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ 模型加载成功")
        
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
        
        # 生成测试数据
        test_images = test_dataset.generate_images(0.1)
        print(f"生成了 {len(test_images)} 个测试样本")
        
        if len(test_images) == 0:
            print("❌ 没有生成测试数据")
            return
        
        # 准备测试数据 - 使用修复后的方法
        image_tensors = []
        for img in test_images[:3]:  # 只测试前3个样本
            img_tensor = torch.FloatTensor(img[0])  # img[0] 是numpy数组，形状为 [height, width]
            image_tensors.append(img_tensor)
        
        images = torch.stack(image_tensors)  # [batch_size, height, width]
        labels = torch.LongTensor([img[1] for img in test_images[:3]]) if setting.TRAIN.LABEL == 'RET5' else torch.LongTensor([img[2] for img in test_images[:3]])
        
        print(f"输入张量形状: {images.shape}")
        print(f"期望形状: [batch_size, height, width] = [3, 32, 15]")
        
        # 进行预测
        with torch.no_grad():
            outputs = model(images)
            print(f"模型输出形状: {outputs.shape}")
            
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            print(f"预测结果: {predictions.tolist()}")
            print(f"真实标签: {labels.tolist()}")
            
            # 计算准确率
            accuracy = (predictions == labels).float().mean().item()
            print(f"准确率: {accuracy:.4f}")
        
        print("✅ 修复验证成功！")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_model()

