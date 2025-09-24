#!/usr/bin/env python3
"""
CNN图像生成可视化脚本
用于可视化生成的OHLCV图像数据
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from __init__ import *
import utils as _U
import dataset as _D

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return _U.Dict2ObjParser(config).parse()

def generate_sample_images(config_path, num_samples=1):
    """生成样本图像数据"""
    print(f"加载配置文件: {config_path}")
    setting = load_config(config_path)
    
    print("初始化数据集...")
    dataset = _D.ImageDataSet(
        win_size=setting.DATASET.LOOKBACK_WIN,
        start_date=setting.DATASET.START_DATE,
        end_date=setting.DATASET.END_DATE,
        mode='train',
        label=setting.TRAIN.LABEL,
        indicators=setting.DATASET.INDICATORS,
        show_volume=setting.DATASET.SHOW_VOLUME,
        parallel_num=setting.DATASET.PARALLEL_NUM
    )
    
    print("生成图像数据...")
    image_set = dataset.generate_images(setting.DATASET.SAMPLE_RATE)
    
    print(f"生成了 {len(image_set)} 张图像")
    
    # 返回前num_samples张图像
    return image_set[:num_samples], setting

def plot_image(image_data, setting, save_path=None):
    """绘制单张图像"""
    image, label_ret5, label_ret20, label_ret60 = image_data
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始图像
    ax1.imshow(image, cmap='gray', aspect='auto')
    ax1.set_title(f'OHLCV图像 (SHOW_VOLUME={setting.DATASET.SHOW_VOLUME})')
    ax1.set_xlabel('时间 (像素)')
    ax1.set_ylabel('价格/成交量 (像素)')
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 图像统计信息
    ax2.text(0.1, 0.9, f'图像尺寸: {image.shape}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.8, f'时间窗口: {setting.DATASET.LOOKBACK_WIN}天', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.7, f'包含成交量: {setting.DATASET.SHOW_VOLUME}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.6, f'标签 (RET5): {label_ret5}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.5, f'标签 (RET20): {label_ret20}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.4, f'标签 (RET60): {label_ret60}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.3, f'非零像素数: {np.count_nonzero(image)}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.2, f'图像最大值: {np.max(image):.1f}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.1, f'图像最小值: {np.min(image):.1f}', transform=ax2.transAxes, fontsize=12)
    
    # 设置信息面板
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('图像信息')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()
    
    return fig

def plot_multiple_images(image_set, setting, num_images=4, save_path=None):
    """绘制多张图像对比"""
    num_images = min(num_images, len(image_set))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_images):
        image, label_ret5, label_ret20, label_ret60 = image_set[i]
        
        axes[i].imshow(image, cmap='gray', aspect='auto')
        axes[i].set_title(f'图像 {i+1} - RET5: {label_ret5}, RET20: {label_ret20}')
        axes[i].set_xlabel('时间 (像素)')
        axes[i].set_ylabel('价格/成交量 (像素)')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(num_images, 4):
        axes[i].axis('off')
    
    plt.suptitle(f'OHLCV图像样本 (SHOW_VOLUME={setting.DATASET.SHOW_VOLUME})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多张图像已保存到: {save_path}")
    
    plt.show()
    
    return fig

def analyze_image_structure(image, setting):
    """分析图像结构"""
    print("\n=== 图像结构分析 ===")
    print(f"图像尺寸: {image.shape}")
    print(f"时间窗口: {setting.DATASET.LOOKBACK_WIN}天")
    print(f"包含成交量: {setting.DATASET.SHOW_VOLUME}")
    
    if setting.DATASET.SHOW_VOLUME:
        if setting.DATASET.LOOKBACK_WIN == 5:
            print("价格区域: 行 7-31 (25行)")
            print("成交量区域: 行 0-5 (6行)")
        else:  # 20天窗口
            print("价格区域: 行 13-63 (51行)")
            print("成交量区域: 行 0-11 (12行)")
    else:
        print("仅包含价格信息，无成交量")
    
    print(f"非零像素数: {np.count_nonzero(image)}")
    print(f"总像素数: {image.size}")
    print(f"填充率: {np.count_nonzero(image)/image.size*100:.2f}%")

def main():
    """主函数"""
    # 默认配置文件路径
    config_path = "configs/I5R5/I5R5_93-00_train.yml"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    try:
        # 生成样本图像
        image_set, setting = generate_sample_images(config_path, num_samples=4)
        
        if len(image_set) == 0:
            print("错误: 没有生成任何图像")
            return
        
        # 分析第一张图像结构
        first_image, _, _, _ = image_set[0]
        analyze_image_structure(first_image, setting)
        
        # 绘制第一张图像
        print("\n绘制第一张图像...")
        plot_image(image_set[0], setting, save_path="first_image_sample.png")
        
        # 绘制多张图像对比
        if len(image_set) > 1:
            print("\n绘制多张图像对比...")
            plot_multiple_images(image_set, setting, save_path="multiple_images_sample.png")
        
        print("\n可视化完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
