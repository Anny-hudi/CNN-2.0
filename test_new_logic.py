#!/usr/bin/env python3
"""
测试新的图像生成逻辑
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator

def test_data_processor():
    """测试数据处理器"""
    print("测试数据处理器...")
    
    try:
        processor = StockDataProcessor(data_fraction=0.1)  # 使用10%数据测试
        processor.load_data()
        
        print(f"加载的股票代码: {list(processor.data.keys())}")
        
        # 测试第一个股票的数据处理
        if processor.data:
            first_symbol = list(processor.data.keys())[0]
            print(f"测试股票: {first_symbol}")
            
            sequences, labels, dates = processor.get_processed_data(first_symbol, 20, 20)
            print(f"生成了 {len(sequences)} 个序列")
            print(f"标签分布: {np.bincount(labels)}")
            
            return True
        else:
            print("没有加载到任何数据")
            return False
            
    except Exception as e:
        print(f"数据处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_generator():
    """测试图像生成器"""
    print("\n测试图像生成器...")
    
    try:
        # 创建测试数据
        import pandas as pd
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(30) * 2,
            'high': 102 + np.random.randn(30) * 2,
            'low': 98 + np.random.randn(30) * 2,
            'close': 100 + np.random.randn(30) * 2,
            'volume': np.random.randint(1000, 10000, 30)
        })
        
        # 添加调整后价格
        test_data['Adj_Open'] = test_data['open']
        test_data['Adj_High'] = test_data['high']
        test_data['Adj_Low'] = test_data['low']
        test_data['Adj_Close_calc'] = test_data['close']
        
        # 测试20天窗口
        generator = OHLCImageGenerator(20)
        image = generator.generate_image(test_data)
        
        print(f"生成的图像尺寸: {image.shape}")
        print(f"图像数据类型: {image.dtype}")
        print(f"图像值范围: {image.min()} - {image.max()}")
        print(f"非零像素数: {np.count_nonzero(image)}")
        
        # 保存测试图像
        plt.figure(figsize=(10, 4))
        plt.imshow(image, cmap='gray', aspect='auto')
        plt.title('测试生成的OHLC图像')
        plt.xlabel('时间（像素）')
        plt.ylabel('价格/成交量（像素）')
        plt.tight_layout()
        plt.savefig('test_generated_image.png', dpi=150)
        plt.close()
        print("测试图像已保存为 test_generated_image.png")
        
        return True
        
    except Exception as e:
        print(f"图像生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试集成功能"""
    print("\n测试集成功能...")
    
    try:
        # 测试数据集类
        from dataset import ImageDataSet
        
        # 创建一个小规模的数据集进行测试
        dataset = ImageDataSet(
            win_size=20,
            start_date=20230101,
            end_date=20231231,
            mode='train',
            label='RET20',
            indicators=[],
            show_volume=True,
            parallel_num=1
        )
        
        print("数据集初始化成功")
        
        # 生成少量图像进行测试
        image_set = dataset.generate_images(sample_rate=0.01)  # 只生成1%的数据
        
        print(f"生成了 {len(image_set)} 张图像")
        
        if len(image_set) > 0:
            # 检查第一张图像
            first_image_data = image_set[0]
            image, ret5, ret20, ret60 = first_image_data
            
            print(f"第一张图像尺寸: {image.shape}")
            print(f"标签: RET5={ret5}, RET20={ret20}, RET60={ret60}")
            
            # 保存第一张图像
            plt.figure(figsize=(10, 4))
            plt.imshow(image.squeeze(), cmap='gray', aspect='auto')
            plt.title('集成测试生成的图像')
            plt.xlabel('时间（像素）')
            plt.ylabel('价格/成交量（像素）')
            plt.tight_layout()
            plt.savefig('test_integration_image.png', dpi=150)
            plt.close()
            print("集成测试图像已保存为 test_integration_image.png")
        
        return True
        
    except Exception as e:
        print(f"集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_60day_images():
    """专门测试60天窗口图像生成"""
    print("\n测试60天窗口图像生成...")
    
    try:
        # 测试数据集类 - 60天窗口
        from dataset import ImageDataSet
        
        # 创建60天窗口的数据集
        dataset = ImageDataSet(
            win_size=60,
            start_date=20230101,
            end_date=20231231,
            mode='train',
            label='RET60',
            indicators=[],
            show_volume=True,
            parallel_num=1
        )
        
        print("60天数据集初始化成功")
        print(f"图像尺寸: {dataset.image_size}")
        
        # 生成少量图像进行测试
        image_set = dataset.generate_images(sample_rate=0.01)  # 只生成1%的数据
        
        print(f"生成了 {len(image_set)} 张60天图像")
        
        if len(image_set) > 0:
            # 检查第一张图像
            first_image_data = image_set[0]
            image, ret5, ret20, ret60 = first_image_data
            
            print(f"第一张60天图像尺寸: {image.shape}")
            print(f"标签: RET5={ret5}, RET20={ret20}, RET60={ret60}")
            
            # 打印图像详细信息
            print(f"图像数据类型: {image.dtype}")
            print(f"图像值范围: {image.min():.3f} - {image.max():.3f}")
            print(f"非零像素数: {np.count_nonzero(image)}")
            print(f"总像素数: {image.size}")
            print(f"填充率: {np.count_nonzero(image)/image.size*100:.2f}%")
            
            # 保存第一张60天图像
            plt.figure(figsize=(15, 6))
            plt.imshow(image.squeeze(), cmap='gray', aspect='auto')
            plt.title('60天窗口OHLC图像 - 第一张样本')
            plt.xlabel('时间（像素）')
            plt.ylabel('价格/成交量（像素）')
            plt.tight_layout()
            plt.savefig('test_60day_first_image.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("60天第一张图像已保存为 test_60day_first_image.png")
            
            # 如果有更多图像，也保存几张对比
            if len(image_set) > 1:
                num_show = min(4, len(image_set))
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(num_show):
                    img_data = image_set[i]
                    img, r5, r20, r60 = img_data
                    axes[i].imshow(img.squeeze(), cmap='gray', aspect='auto')
                    axes[i].set_title(f'60天图像 {i+1} - RET60: {r60}')
                    axes[i].set_xlabel('时间（像素）')
                    axes[i].set_ylabel('价格/成交量（像素）')
                
                # 隐藏多余的子图
                for i in range(num_show, 4):
                    axes[i].axis('off')
                
                plt.suptitle('60天窗口OHLC图像样本对比', fontsize=16)
                plt.tight_layout()
                plt.savefig('test_60day_multiple_images.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"60天多张图像对比已保存为 test_60day_multiple_images.png")
        
        return True
        
    except Exception as e:
        print(f"60天图像测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试新的图像生成逻辑...")
    print("=" * 50)
    
    # 检查数据目录
    if not os.path.exists('data'):
        print("错误: 未找到data目录，请确保数据文件存在")
        return
    
    # 运行测试
    tests = [
        ("数据处理器", test_data_processor),
        ("图像生成器", test_image_generator),
        ("集成功能", test_integration),
        ("60天图像生成", test_60day_images)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # 打印测试结果
    print(f"\n{'='*50}")
    print("测试结果总结:")
    print(f"{'='*50}")
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 所有测试通过！新的图像生成逻辑工作正常。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
