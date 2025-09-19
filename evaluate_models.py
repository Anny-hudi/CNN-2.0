#!/usr/bin/env python3
"""
I5R5_OHLCV模型统一性能评估脚本
测试所有I5R5_OHLCV模型的性能和预测效果
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from __init__ import *
import utils as _U
import model as _M
import dataset as _D

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_dir="configs/I5R5", model_dir="models/I5R5_OHLCV"):
        self.config_dir = config_dir
        self.model_dir = model_dir
        self.results = []
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return _U.Dict2ObjParser(config).parse()
    
    def load_model(self, model_path, setting):
        """加载训练好的模型"""
        model = _M.CNN5d()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def prepare_test_data(self, setting):
        """准备测试数据"""
        print(f"准备测试数据: {setting.DATASET.START_DATE} - {setting.DATASET.END_DATE}")
        
        # 创建测试数据集
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
        
        # 生成测试图像
        test_images = test_dataset.generate_images(1.0)  # 使用所有数据
        
        if len(test_images) == 0:
            print(f"警告: 没有生成测试数据")
            return None, None, None
        
        # 转换为PyTorch张量
        # 模型期望输入格式: [batch_size, height, width]，模型内部会添加通道维度
        image_tensors = []
        for img in test_images:
            img_tensor = torch.FloatTensor(img[0])  # img[0] 是numpy数组，形状为 [height, width]
            image_tensors.append(img_tensor)
        
        images = torch.stack(image_tensors)  # [batch_size, height, width]
        labels_ret5 = torch.LongTensor([img[1] for img in test_images])
        labels_ret20 = torch.LongTensor([img[2] for img in test_images])
        
        # 根据标签选择对应的标签
        if setting.TRAIN.LABEL == 'RET5':
            labels = labels_ret5
        else:
            labels = labels_ret20
            
        return images, labels, test_images
    
    def evaluate_model(self, config_file, model_file):
        """评估单个模型"""
        print(f"\n=== 评估模型: {config_file} ===")
        
        try:
            # 加载配置
            config_path = os.path.join(self.config_dir, config_file)
            setting = self.load_config(config_path)
            
            # 检查模型文件是否存在
            model_path = os.path.join(self.model_dir, model_file)
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return None
            
            # 加载模型
            model = self.load_model(model_path, setting)
            
            # 准备测试数据
            images, labels, raw_data = self.prepare_test_data(setting)
            if images is None:
                return None
            
            # 进行预测
            with torch.no_grad():
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
            
            # 计算性能指标
            y_true = labels.numpy()
            y_pred = predictions.numpy()
            y_prob = probabilities[:, 1].numpy()  # 正类概率
            
            # 基本指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # AUC (如果数据平衡)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # 计算特异性
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # 数据统计
            total_samples = len(y_true)
            positive_samples = np.sum(y_true)
            negative_samples = total_samples - positive_samples
            
            result = {
                'config_file': config_file,
                'model_file': model_file,
                'time_period': f"{setting.DATASET.START_DATE}-{setting.DATASET.END_DATE}",
                'test_period': f"{setting.TEST.START_DATE}-{setting.TEST.END_DATE}",
                'label': setting.TRAIN.LABEL,
                'total_samples': total_samples,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'specificity': specificity,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            
            print(f"测试样本数: {total_samples}")
            print(f"正样本: {positive_samples}, 负样本: {negative_samples}")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"特异性: {specificity:.4f}")
            
            return result
            
        except Exception as e:
            print(f"评估模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_all_models(self):
        """评估所有模型"""
        print("开始评估所有I5R5_OHLCV模型...")
        
        # 获取所有配置文件
        config_files = [f for f in os.listdir(self.config_dir) if f.endswith('.yml')]
        config_files.sort()
        
        for config_file in config_files:
            # 从配置文件名生成模型文件名
            model_file = config_file.replace('.yml', '.tar').replace('I5R5_', 'I5R5_OHLCV_')
            
            result = self.evaluate_model(config_file, model_file)
            if result:
                self.results.append(result)
        
        print(f"\n完成评估，共评估了 {len(self.results)} 个模型")
        return self.results
    
    def create_summary_report(self):
        """创建汇总报告"""
        if not self.results:
            print("没有评估结果")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存详细结果
        df.to_csv('model_evaluation_results.csv', index=False)
        print("详细结果已保存到: model_evaluation_results.csv")
        
        # 打印汇总统计
        print("\n=== 模型性能汇总 ===")
        print(f"平均准确率: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"平均精确率: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
        print(f"平均召回率: {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
        print(f"平均F1分数: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
        print(f"平均AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        
        # 找出最佳模型
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        best_f1 = df.loc[df['f1_score'].idxmax()]
        best_auc = df.loc[df['auc'].idxmax()]
        
        print(f"\n最佳准确率模型: {best_accuracy['config_file']} (准确率: {best_accuracy['accuracy']:.4f})")
        print(f"最佳F1分数模型: {best_f1['config_file']} (F1: {best_f1['f1_score']:.4f})")
        print(f"最佳AUC模型: {best_auc['config_file']} (AUC: {best_auc['auc']:.4f})")
        
        return df
    
    def plot_performance_comparison(self, df):
        """绘制性能对比图"""
        if df.empty:
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 提取时间周期用于x轴标签
        time_periods = [f"{row['time_period']}" for _, row in df.iterrows()]
        
        # 1. 准确率对比
        axes[0, 0].bar(range(len(df)), df['accuracy'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('模型准确率对比')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(time_periods, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 精确率对比
        axes[0, 1].bar(range(len(df)), df['precision'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('模型精确率对比')
        axes[0, 1].set_ylabel('精确率')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(time_periods, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 召回率对比
        axes[0, 2].bar(range(len(df)), df['recall'], color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('模型召回率对比')
        axes[0, 2].set_ylabel('召回率')
        axes[0, 2].set_xticks(range(len(df)))
        axes[0, 2].set_xticklabels(time_periods, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. F1分数对比
        axes[1, 0].bar(range(len(df)), df['f1_score'], color='gold', alpha=0.7)
        axes[1, 0].set_title('模型F1分数对比')
        axes[1, 0].set_ylabel('F1分数')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(time_periods, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. AUC对比
        axes[1, 1].bar(range(len(df)), df['auc'], color='plum', alpha=0.7)
        axes[1, 1].set_title('模型AUC对比')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(time_periods, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 综合性能雷达图（选择前5个模型）
        top_models = df.nlargest(5, 'accuracy')
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax_radar = axes[1, 2]
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (_, model) in enumerate(top_models.iterrows()):
            values = [model[metric] for metric in metrics]
            values += values[:1]  # 闭合图形
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=f"模型{i+1}", color=colors[i])
            ax_radar.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(['准确率', '精确率', '召回率', 'F1分数', 'AUC'])
        ax_radar.set_title('前5个模型综合性能对比')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax_radar.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("性能对比图已保存到: model_performance_comparison.png")
    
    def create_detailed_analysis(self, df):
        """创建详细分析报告"""
        if df.empty:
            return
        
        print("\n=== 详细分析报告 ===")
        
        # 按时间周期分析
        print("\n1. 按训练时间周期分析:")
        for period in df['time_period'].unique():
            period_data = df[df['time_period'] == period]
            print(f"   {period}: 平均准确率 {period_data['accuracy'].mean():.4f}")
        
        # 按测试时间周期分析
        print("\n2. 按测试时间周期分析:")
        for period in df['test_period'].unique():
            period_data = df[df['test_period'] == period]
            print(f"   {period}: 平均准确率 {period_data['accuracy'].mean():.4f}")
        
        # 性能分布分析
        print("\n3. 性能分布分析:")
        print(f"   准确率范围: {df['accuracy'].min():.4f} - {df['accuracy'].max():.4f}")
        print(f"   F1分数范围: {df['f1_score'].min():.4f} - {df['f1_score'].max():.4f}")
        print(f"   AUC范围: {df['auc'].min():.4f} - {df['auc'].max():.4f}")
        
        # 稳定性分析
        print("\n4. 模型稳定性分析:")
        print(f"   准确率标准差: {df['accuracy'].std():.4f}")
        print(f"   F1分数标准差: {df['f1_score'].std():.4f}")
        print(f"   AUC标准差: {df['auc'].std():.4f}")
        
        # 推荐模型
        print("\n5. 推荐模型:")
        # 综合评分 (准确率 + F1 + AUC)
        df['综合评分'] = (df['accuracy'] + df['f1_score'] + df['auc']) / 3
        top_3 = df.nlargest(3, '综合评分')
        for i, (_, model) in enumerate(top_3.iterrows(), 1):
            print(f"   第{i}名: {model['config_file']} (综合评分: {model['综合评分']:.4f})")

def main():
    """主函数"""
    print("I5R5_OHLCV模型统一性能评估")
    print("=" * 50)
    
    # 创建评估器
    evaluator = ModelEvaluator()
    
    # 评估所有模型
    results = evaluator.evaluate_all_models()
    
    if not results:
        print("没有成功评估任何模型")
        return
    
    # 创建汇总报告
    df = evaluator.create_summary_report()
    
    # 绘制性能对比图
    evaluator.plot_performance_comparison(df)
    
    # 创建详细分析
    evaluator.create_detailed_analysis(df)
    
    print("\n评估完成！")
    print("生成的文件:")
    print("- model_evaluation_results.csv: 详细评估结果")
    print("- model_performance_comparison.png: 性能对比图")

if __name__ == "__main__":
    main()
