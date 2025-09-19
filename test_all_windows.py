#!/usr/bin/env python3
"""
测试所有时间窗口的图像生成（5天、20天、60天）
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageDataSet

def test_window(window_days, label):
    """测试指定时间窗口的图像生成"""
    print(f"\n{'='*20} 测试{window_days}天窗口 {'='*20}")
    
    try:
        # 创建数据集
        print(f"初始化{window_days}天数据集...")
        dataset = ImageDataSet(
            win_size=window_days,
            start_date=19930101,
            end_date=20191231,
            mode='train',
            label=label,
            indicators=[],
            show_volume=True,
            parallel_num=1
        )
        
        print(f"图像尺寸: {dataset.image_size}")
        print("数据集初始化成功")
        
        # 生成少量图像进行测试
        print("生成图像数据...")
        image_set = dataset.generate_images(sample_rate=0.01)  # 只生成1%的数据
        
        print(f"生成了 {len(image_set)} 张{window_days}天图像")
        
        if len(image_set) > 0:
            # 检查第一张图像
            first_image_data = image_set[0]
            image, ret5, ret20, ret60 = first_image_data
            
            print(f"\n第一张{window_days}天图像信息:")
            print(f"  图像尺寸: {image.shape}")
            print(f"  标签: RET5={ret5}, RET20={ret20}, RET60={ret60}")
            print(f"  图像数据类型: {image.dtype}")
            print(f"  图像值范围: {image.min():.3f} - {image.max():.3f}")
            print(f"  非零像素数: {np.count_nonzero(image)}")
            print(f"  总像素数: {image.size}")
            print(f"  填充率: {np.count_nonzero(image)/image.size*100:.2f}%")
            
            # 保存第一张图像
            print(f"\n保存第一张{window_days}天图像...")
            fig_width = 15 if window_days == 60 else 12 if window_days == 20 else 10
            plt.figure(figsize=(fig_width, 5))
            plt.imshow(image.squeeze(), cmap='gray', aspect='auto')
            plt.title(f'{window_days}天窗口OHLC图像 - 第一张样本')
            plt.xlabel('时间（像素）')
            plt.ylabel('价格/成交量（像素）')
            plt.tight_layout()
            plt.savefig(f'test_{window_days}day_first_image.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ {window_days}天第一张图像已保存为 test_{window_days}day_first_image.png")
            
            return True, len(image_set), image.shape
            
        else:
            print(f"❌ 没有生成任何{window_days}天图像")
            return False, 0, None
        
    except Exception as e:
        print(f"❌ {window_days}天图像测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, None

def main():
    """主函数"""
    print("所有时间窗口图像生成测试")
    print("=" * 60)
    
    # 检查数据目录
    if not os.path.exists('data'):
        print("❌ 错误: 未找到data目录，请确保数据文件存在")
        return
    
    # 测试配置
    test_configs = [
        (5, 'RET5'),
        (20, 'RET20'),
        (60, 'RET60')
    ]
    
    results = []
    
    # 运行所有测试
    for window_days, label in test_configs:
        success, num_images, image_shape = test_window(window_days, label)
        results.append((window_days, success, num_images, image_shape))
    
    # 打印测试结果总结
    print(f"\n{'='*60}")
    print("测试结果总结:")
    print(f"{'='*60}")
    print(f"{'窗口':^8}{'状态':^8}{'图像数':^10}{'图像尺寸':^15}")
    print("-" * 60)
    
    all_success = True
    for window_days, success, num_images, image_shape in results:
        status = "✓ 成功" if success else "✗ 失败"
        shape_str = str(image_shape) if image_shape else "N/A"
        print(f"{window_days}天:^8{status:^8}{num_images:^10}{shape_str:^15}")
        if not success:
            all_success = False
    
    print(f"\n{'='*60}")
    if all_success:
        print("🎉 所有时间窗口测试成功完成！")
        print("\n生成的图像文件:")
        for window_days, _, _, _ in results:
            print(f"   - test_{window_days}day_first_image.png")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
