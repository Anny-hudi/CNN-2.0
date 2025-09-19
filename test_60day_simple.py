#!/usr/bin/env python3
"""
简单测试60天窗口图像生成
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageDataSet

def main():
    """主函数"""
    print("60天窗口图像生成测试")
    print("=" * 50)
    
    # 检查数据目录
    if not os.path.exists('data'):
        print("❌ 错误: 未找到data目录，请确保数据文件存在")
        return
    
    try:
        # 创建60天窗口的数据集
        print("初始化60天数据集...")
        dataset = ImageDataSet(
            win_size=60,
            start_date=19930101,
            end_date=20191231,
            mode='train',
            label='RET60',
            indicators=[],
            show_volume=True,
            parallel_num=1
        )
        
        print(f"图像尺寸: {dataset.image_size}")
        print("数据集初始化成功")
        
        # 生成少量图像进行测试
        print("\n生成图像数据...")
        image_set = dataset.generate_images(sample_rate=0.01)  # 只生成1%的数据
        
        print(f"生成了 {len(image_set)} 张60天图像")
        
        if len(image_set) > 0:
            # 检查第一张图像
            first_image_data = image_set[0]
            image, ret5, ret20, ret60 = first_image_data
            
            print(f"\n第一张60天图像信息:")
            print(f"  图像尺寸: {image.shape}")
            print(f"  标签: RET5={ret5}, RET20={ret20}, RET60={ret60}")
            print(f"  图像数据类型: {image.dtype}")
            print(f"  图像值范围: {image.min():.3f} - {image.max():.3f}")
            print(f"  非零像素数: {np.count_nonzero(image)}")
            print(f"  总像素数: {image.size}")
            print(f"  填充率: {np.count_nonzero(image)/image.size*100:.2f}%")
            
            # 保存第一张60天图像
            print("\n保存第一张60天图像...")
            plt.figure(figsize=(15, 6))
            plt.imshow(image.squeeze(), cmap='gray', aspect='auto')
            plt.title('60天窗口OHLC图像 - 第一张样本')
            plt.xlabel('时间（像素）')
            plt.ylabel('价格/成交量（像素）')
            plt.tight_layout()
            plt.savefig('test_60day_first_image.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 60天第一张图像已保存为 test_60day_first_image.png")
            
            print(f"\n🎉 60天图像测试完成！")
            print(f"   生成了 {len(image_set)} 张图像")
            print(f"   图像尺寸: {image.shape}")
            print(f"   保存了图像文件: test_60day_first_image.png")
            
        else:
            print("❌ 没有生成任何图像")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 60天图像测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
