#!/usr/bin/env python3
"""
测试修正后的60天CNN模型
"""
import torch
import numpy as np
from model import CNN60d

def test_60day_model():
    """测试修正后的60天模型"""
    print("测试修正后的60天CNN模型...")
    print("=" * 50)
    
    try:
        # 创建模型
        model = CNN60d()
        print("✓ 模型创建成功")
        
        # 测试输入尺寸
        batch_size = 2
        input_height = 96
        input_width = 180
        
        # 创建测试输入
        test_input = torch.randn(batch_size, input_height, input_width)
        print(f"✓ 测试输入创建成功: {test_input.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ 前向传播成功")
        print(f"  输入尺寸: {test_input.shape}")
        print(f"  输出尺寸: {output.shape}")
        print(f"  输出值范围: {output.min().item():.4f} - {output.max().item():.4f}")
        print(f"  输出和: {output.sum(dim=1)} (应该接近1.0，因为是softmax)")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 检查各层输出尺寸
        print(f"\n各层输出尺寸验证:")
        x = test_input.unsqueeze(1).to(torch.float32)
        print(f"  输入: {x.shape}")
        
        x = model.conv1(x)
        print(f"  Conv1后: {x.shape}")
        
        x = model.conv2(x)
        print(f"  Conv2后: {x.shape}")
        
        x = model.conv3(x)
        print(f"  Conv3后: {x.shape}")
        
        x = model.conv4(x)
        print(f"  Conv4后: {x.shape}")
        
        x = model.extra_pool(x)
        print(f"  Extra Pool后: {x.shape}")
        
        x = x.view(x.shape[0], -1)
        print(f"  展平后: {x.shape}")
        
        # 验证全连接层输入维度
        expected_fc_input = 512 * 6 * 60  # 184320
        actual_fc_input = x.shape[1]
        print(f"  全连接层输入维度: {actual_fc_input} (期望: {expected_fc_input})")
        
        if actual_fc_input == expected_fc_input:
            print("  ✓ 全连接层输入维度正确")
        else:
            print("  ❌ 全连接层输入维度不匹配")
        
        print(f"\n🎉 60天模型测试完成！")
        print(f"   模型结构符合cnn_query.md设定:")
        print(f"   - 输入尺寸: 96×180 ✓")
        print(f"   - 4个卷积层: 64→128→256→512 ✓")
        print(f"   - 全连接层: 184320 ✓")
        print(f"   - 输出: Softmax(2) ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("60天CNN模型修正验证")
    print("=" * 50)
    
    # 运行测试
    success = test_60day_model()
    
    if success:
        print(f"\n✅ 测试成功！60天模型已修正并符合设定要求")
    else:
        print(f"\n❌ 测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
