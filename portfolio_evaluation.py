"""
投资组合评估模块 - 实现论文要求的十分位组合和夏普比率分析
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class PortfolioEvaluator:
    """投资组合评估器"""
    
    def __init__(self, risk_free_rate=0.02):
        """
        初始化投资组合评估器
        
        Args:
            risk_free_rate: 无风险利率，默认2%
        """
        self.risk_free_rate = risk_free_rate
        self.results = {}
        
    def create_decile_portfolios(self, predictions, returns, dates=None):
        """
        创建十分位投资组合
        
        Args:
            predictions: 模型预测概率 [N, 2] 或 [N]
            returns: 实际收益率 [N]
            dates: 日期列表 [N]
            
        Returns:
            portfolio_results: 投资组合结果字典
        """
        # 确保输入是numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        
        # 如果是二分类概率，取上涨概率
        if predictions.ndim == 2:
            up_prob = predictions[:, 1]  # 上涨概率
        else:
            up_prob = predictions
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates if dates is not None else range(len(predictions)),
            'prediction': up_prob,
            'return': returns
        })
        
        # 按预测概率排序并分组
        df = df.sort_values('prediction').reset_index(drop=True)
        df['decile'] = pd.qcut(df['prediction'], q=10, labels=False, duplicates='drop')
        
        # 计算每个十分位的统计量
        decile_stats = df.groupby('decile').agg({
            'return': ['mean', 'std', 'count'],
            'prediction': ['mean', 'min', 'max']
        }).round(6)
        
        # 计算夏普比率
        decile_returns = df.groupby('decile')['return'].mean()
        decile_volatilities = df.groupby('decile')['return'].std()
        sharpe_ratios = (decile_returns - self.risk_free_rate/252) / decile_volatilities  # 日夏普比率
        
        # 年化夏普比率
        annualized_sharpe = sharpe_ratios * np.sqrt(252)
        
        # 构建结果
        portfolio_results = {
            'decile_stats': decile_stats,
            'decile_returns': decile_returns,
            'decile_volatilities': decile_volatilities,
            'sharpe_ratios': sharpe_ratios,
            'annualized_sharpe': annualized_sharpe,
            'data': df
        }
        
        return portfolio_results
    
    def create_hl_strategy(self, portfolio_results):
        """
        创建H-L策略（多空策略）
        
        Args:
            portfolio_results: 十分位投资组合结果
            
        Returns:
            hl_results: H-L策略结果
        """
        decile_returns = portfolio_results['decile_returns']
        
        # H-L策略：做多第10分位，做空第1分位
        long_return = decile_returns.iloc[-1]  # 第10分位（最高预测概率）
        short_return = decile_returns.iloc[0]  # 第1分位（最低预测概率）
        
        # H-L组合收益率
        hl_return = long_return - short_return
        
        # 计算H-L组合的波动率（假设两个分位独立）
        long_vol = portfolio_results['decile_volatilities'].iloc[-1]
        short_vol = portfolio_results['decile_volatilities'].iloc[0]
        hl_volatility = np.sqrt(long_vol**2 + short_vol**2)  # 假设独立
        
        # H-L夏普比率
        hl_sharpe = hl_return / hl_volatility if hl_volatility > 0 else 0
        hl_annualized_sharpe = hl_sharpe * np.sqrt(252)
        
        hl_results = {
            'long_return': long_return,
            'short_return': short_return,
            'hl_return': hl_return,
            'hl_volatility': hl_volatility,
            'hl_sharpe': hl_sharpe,
            'hl_annualized_sharpe': hl_annualized_sharpe,
            'long_decile': 10,
            'short_decile': 1
        }
        
        return hl_results
    
    def evaluate_model_performance(self, model, test_data, test_labels, test_dates=None):
        """
        评估模型性能并创建投资组合
        
        Args:
            model: 训练好的模型
            test_data: 测试数据 [N, H, W]
            test_labels: 测试标签 [N]
            test_dates: 测试日期 [N]
            
        Returns:
            evaluation_results: 评估结果
        """
        model.eval()
        
        # 确保数据和模型在同一设备上
        device = next(model.parameters()).device
        
        with torch.no_grad():
            # 获取预测概率
            if isinstance(test_data, torch.Tensor):
                test_data = test_data.to(device)
                predictions = model(test_data)
            else:
                test_data = torch.FloatTensor(test_data).to(device)
                predictions = model(test_data)
            
            # 转换为numpy
            predictions = predictions.cpu().numpy()
            labels = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels
            
            # 计算基本指标
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == labels)
            
            # 计算AUC
            up_prob = predictions[:, 1]
            auc = roc_auc_score(labels, up_prob)
            
            # 创建十分位投资组合
            portfolio_results = self.create_decile_portfolios(predictions, labels, test_dates)
            
            # 创建H-L策略
            hl_results = self.create_hl_strategy(portfolio_results)
            
            evaluation_results = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': predictions,
                'portfolio_results': portfolio_results,
                'hl_results': hl_results
            }
            
            return evaluation_results
    
    def plot_decile_analysis(self, portfolio_results, save_path=None):
        """
        绘制十分位分析图
        
        Args:
            portfolio_results: 十分位投资组合结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        decile_returns = portfolio_results['decile_returns']
        annualized_sharpe = portfolio_results['annualized_sharpe']
        decile_volatilities = portfolio_results['decile_volatilities']
        
        # 1. 十分位收益率
        axes[0, 0].bar(range(1, 11), decile_returns, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('十分位组合平均收益率')
        axes[0, 0].set_xlabel('十分位')
        axes[0, 0].set_ylabel('平均收益率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 年化夏普比率
        axes[0, 1].bar(range(1, 11), annualized_sharpe, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('十分位组合年化夏普比率')
        axes[0, 1].set_xlabel('十分位')
        axes[0, 1].set_ylabel('年化夏普比率')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 收益率vs夏普比率散点图
        axes[1, 0].scatter(decile_returns, annualized_sharpe, s=100, alpha=0.7, c=range(1, 11), cmap='viridis')
        axes[1, 0].set_title('收益率 vs 夏普比率')
        axes[1, 0].set_xlabel('平均收益率')
        axes[1, 0].set_ylabel('年化夏普比率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加十分位标签
        for i, (ret, sharpe) in enumerate(zip(decile_returns, annualized_sharpe)):
            axes[1, 0].annotate(f'{i+1}', (ret, sharpe), xytext=(5, 5), textcoords='offset points')
        
        # 4. 波动率分析
        axes[1, 1].bar(range(1, 11), decile_volatilities, color='salmon', alpha=0.7)
        axes[1, 1].set_title('十分位组合波动率')
        axes[1, 1].set_xlabel('十分位')
        axes[1, 1].set_ylabel('波动率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"十分位分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_hl_strategy_analysis(self, hl_results, save_path=None):
        """
        绘制H-L策略分析图
        
        Args:
            hl_results: H-L策略结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 多空收益率对比
        categories = ['做空(第1分位)', '做多(第10分位)', 'H-L策略']
        returns = [hl_results['short_return'], hl_results['long_return'], hl_results['hl_return']]
        colors = ['red', 'green', 'blue']
        
        bars = axes[0].bar(categories, returns, color=colors, alpha=0.7)
        axes[0].set_title('H-L策略收益率分析')
        axes[0].set_ylabel('收益率')
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.4f}', ha='center', va='bottom')
        
        # 2. 夏普比率对比
        sharpe_values = [hl_results['hl_sharpe']] * 3  # 简化显示
        axes[1].bar(['H-L策略'], [hl_results['hl_annualized_sharpe']], color='blue', alpha=0.7)
        axes[1].set_title('H-L策略年化夏普比率')
        axes[1].set_ylabel('年化夏普比率')
        axes[1].grid(True, alpha=0.3)
        
        # 添加数值标签
        axes[1].text(0, hl_results['hl_annualized_sharpe'],
                    f'{hl_results["hl_annualized_sharpe"]:.4f}', 
                    ha='center', va='bottom')
        
        # 3. 风险收益散点图
        axes[2].scatter(hl_results['hl_volatility'], hl_results['hl_return'], 
                       s=200, c='blue', alpha=0.7, label='H-L策略')
        axes[2].set_title('H-L策略风险收益分析')
        axes[2].set_xlabel('波动率')
        axes[2].set_ylabel('收益率')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"H-L策略分析图已保存到: {save_path}")
        
        plt.show()
    
    def generate_portfolio_report(self, evaluation_results, model_name, save_path=None):
        """
        生成投资组合报告
        
        Args:
            evaluation_results: 评估结果
            model_name: 模型名称
            save_path: 保存路径
        """
        portfolio_results = evaluation_results['portfolio_results']
        hl_results = evaluation_results['hl_results']
        
        print(f"\n{'='*60}")
        print(f"{model_name} 投资组合评估报告")
        print(f"{'='*60}")
        
        print(f"模型性能:")
        print(f"  准确率: {evaluation_results['accuracy']:.4f}")
        print(f"  AUC: {evaluation_results['auc']:.4f}")
        
        print(f"\n十分位组合分析:")
        decile_returns = portfolio_results['decile_returns']
        annualized_sharpe = portfolio_results['annualized_sharpe']
        
        for i in range(10):
            print(f"  第{i+1}分位: 收益率={decile_returns.iloc[i]:.4f}, "
                  f"年化夏普比率={annualized_sharpe.iloc[i]:.4f}")
        
        print(f"\nH-L策略分析:")
        print(f"  做多收益率(第10分位): {hl_results['long_return']:.4f}")
        print(f"  做空收益率(第1分位): {hl_results['short_return']:.4f}")
        print(f"  H-L策略收益率: {hl_results['hl_return']:.4f}")
        print(f"  H-L策略波动率: {hl_results['hl_volatility']:.4f}")
        print(f"  H-L策略年化夏普比率: {hl_results['hl_annualized_sharpe']:.4f}")
        
        # 保存详细报告
        if save_path:
            report_data = {
                'model_name': model_name,
                'accuracy': evaluation_results['accuracy'],
                'auc': evaluation_results['auc'],
                'hl_return': hl_results['hl_return'],
                'hl_volatility': hl_results['hl_volatility'],
                'hl_annualized_sharpe': hl_results['hl_annualized_sharpe']
            }
            
            # 添加十分位数据
            for i in range(10):
                report_data[f'decile_{i+1}_return'] = decile_returns.iloc[i]
                report_data[f'decile_{i+1}_sharpe'] = annualized_sharpe.iloc[i]
            
            report_df = pd.DataFrame([report_data])
            report_df.to_csv(save_path, index=False)
            print(f"\n详细报告已保存到: {save_path}")

def main():
    """主函数 - 示例用法"""
    # 这里可以添加示例代码
    pass

if __name__ == '__main__':
    main()
