#!/usr/bin/env python3
"""
快速测试20天窗口图像生成
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageDataSet

def test_20day_quick():
    """快速测试20天图像生成"""
    print("快速测试20天窗口图像生成...")
    print("=" * 50)
    
    try:
        # 创建20天窗口的数据集
        print("初始化20天数据集...")
        dataset = ImageDataSet(
            win_size=20,
            start_date=19930101,
            end_date=20191231,
            mode='train',
            label='RET20',
            indicators=[],
            show_volume=True,
            parallel_num=1
        )
        
        print(f"图像尺寸: {dataset.image_size}")
        print("数据集初始化成功")
        
        # 生成少量图像进行测试
        print("生成图像数据...")
        image_set = dataset.generate_images(sample_rate=0.01)  # 只生成1%的数据
        
        print(f"生成了 {len(image_set)} 张20天图像")
        
        if len(image_set) > 0:
            # 检查第一张图像
            first_image_data = image_set[0]
            image, ret5, ret20, ret60 = first_image_data
            
            print(f"\n第一张20天图像信息:")
            print(f"  图像尺寸: {image.shape}")
            print(f"  标签: RET5={ret5}, RET20={ret20}, RET60={ret60}")
            print(f"  图像数据类型: {image.dtype}")
            print(f"  图像值范围: {image.min():.3f} - {image.max():.3f}")
            print(f"  非零像素数: {np.count_nonzero(image)}")
            print(f"  总像素数: {image.size}")
            print(f"  填充率: {np.count_nonzero(image)/image.size*100:.2f}%")
            
            # 保存第一张20天图像
            print("\n保存第一张20天图像...")
            # 20天图像尺寸是64x60，使用1:1像素比例显示
            fig_width = 60 * 0.1  # 60像素宽度，每像素0.1英寸
            fig_height = 64 * 0.1  # 64像素高度，每像素0.1英寸
            plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(image.squeeze(), cmap='gray', aspect=1)  # aspect=1 保持1:1像素比例
            plt.title('20天窗口OHLC图像 - 第一张样本')
            plt.xlabel('时间（像素）')
            plt.ylabel('价格/成交量（像素）')
            plt.tight_layout()
            plt.savefig('test_20day_first_image.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 20天第一张图像已保存为 test_20day_first_image.png")
            
            # 如果有更多图像，也保存几张对比
            if len(image_set) > 1:
                print(f"\n保存多张20天图像对比...")
                num_show = min(4, len(image_set))
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(num_show):
                    img_data = image_set[i]
                    img, r5, r20, r60 = img_data
                    axes[i].imshow(img.squeeze(), cmap='gray', aspect=1)  # aspect=1 保持1:1像素比例
                    axes[i].set_title(f'20天图像 {i+1} - RET20: {r20}')
                    axes[i].set_xlabel('时间（像素）')
                    axes[i].set_ylabel('价格/成交量（像素）')
                
                # 隐藏多余的子图
                for i in range(num_show, 4):
                    axes[i].axis('off')
                
                plt.suptitle('20天窗口OHLC图像样本对比', fontsize=16)
                plt.tight_layout()
                plt.savefig('test_20day_multiple_images.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ 20天多张图像对比已保存为 test_20day_multiple_images.png")
            
            print(f"\n🎉 20天图像测试完成！")
            print(f"   生成了 {len(image_set)} 张图像")
            print(f"   图像尺寸: {image.shape}")
            print(f"   保存了 {1 + (1 if len(image_set) > 1 else 0)} 个图像文件")
            
        else:
            print("❌ 没有生成任何图像")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 20天图像测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("20天窗口图像生成快速测试")
    print("=" * 50)
    
    # 检查数据目录
    if not os.path.exists('data'):
        print("❌ 错误: 未找到data目录，请确保数据文件存在")
        return
    
    # 运行20天测试
    success = test_20day_quick()
    
    if success:
        print(f"\n✅ 测试成功完成！")
        print(f"   请查看生成的图像文件:")
        print(f"   - test_20day_first_image.png")
        if os.path.exists('test_20day_multiple_images.png'):
            print(f"   - test_20day_multiple_images.png")
    else:
        print(f"\n❌ 测试失败，请检查错误信息")

if __name__ == "__main__":
    main()
