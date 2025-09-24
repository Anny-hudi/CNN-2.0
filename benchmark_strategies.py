"""
基准策略模块 - 实现论文中提到的MOM、STR、WSTR、TREND策略
"""
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class BenchmarkStrategies:
    """基准策略类"""
    
    def __init__(self, risk_free_rate=0.02):
        """
        初始化基准策略
        
        Args:
            risk_free_rate: 无风险利率，默认2%
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_momentum(self, prices, window=20):
        """
        计算动量因子 (MOM)
        
        Args:
            prices: 价格序列
            window: 计算窗口
            
        Returns:
            momentum: 动量因子
        """
        returns = prices.pct_change()
        momentum = returns.rolling(window=window).sum()
        return momentum
    
    def calculate_str(self, prices, window=20):
        """
        计算短期趋势因子 (STR)
        
        Args:
            prices: 价格序列
            window: 计算窗口
            
        Returns:
            str_factor: 短期趋势因子
        """
        # 计算移动平均线
        ma = prices.rolling(window=window).mean()
        
        # 计算价格相对于移动平均线的偏离度
        str_factor = (prices - ma) / ma
        return str_factor
    
    def calculate_wstr(self, prices, window=20):
        """
        计算加权短期趋势因子 (WSTR)
        
        Args:
            prices: 价格序列
            window: 计算窗口
            
        Returns:
            wstr_factor: 加权短期趋势因子
        """
        # 计算加权移动平均线（近期权重更高）
        weights = np.arange(1, window + 1)
        wma = prices.rolling(window=window).apply(
            lambda x: np.average(x, weights=weights), raw=True
        )
        
        # 计算价格相对于加权移动平均线的偏离度
        wstr_factor = (prices - wma) / wma
        return wstr_factor
    
    def calculate_trend(self, prices, short_window=5, long_window=20):
        """
        计算趋势因子 (TREND)
        
        Args:
            prices: 价格序列
            short_window: 短期窗口
            long_window: 长期窗口
            
        Returns:
            trend_factor: 趋势因子
        """
        # 计算短期和长期移动平均线
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        
        # 计算趋势因子
        trend_factor = (short_ma - long_ma) / long_ma
        return trend_factor
    
    def calculate_rsi(self, prices, window=14):
        """
        计算相对强弱指数 (RSI)
        
        Args:
            prices: 价格序列
            window: 计算窗口
            
        Returns:
            rsi: RSI指标
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """
        计算布林带
        
        Args:
            prices: 价格序列
            window: 计算窗口
            num_std: 标准差倍数
            
        Returns:
            bb_upper: 上轨
            bb_lower: 下轨
            bb_middle: 中轨
            bb_width: 布林带宽度
        """
        bb_middle = prices.rolling(window=window).mean()
        bb_std = prices.rolling(window=window).std()
        
        bb_upper = bb_middle + (bb_std * num_std)
        bb_lower = bb_middle - (bb_std * num_std)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        return bb_upper, bb_lower, bb_middle, bb_width
    
    def create_benchmark_signals(self, df, price_col='close'):
        """
        创建所有基准策略信号
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名
            
        Returns:
            signals_df: 包含所有信号的DataFrame
        """
        prices = df[price_col]
        
        signals_df = pd.DataFrame(index=df.index)
        signals_df['date'] = df.get('date', df.index)
        
        # 计算各种因子
        signals_df['MOM'] = self.calculate_momentum(prices, window=20)
        signals_df['STR'] = self.calculate_str(prices, window=20)
        signals_df['WSTR'] = self.calculate_wstr(prices, window=20)
        signals_df['TREND'] = self.calculate_trend(prices, short_window=5, long_window=20)
        signals_df['RSI'] = self.calculate_rsi(prices, window=14)
        
        # 计算布林带
        bb_upper, bb_lower, bb_middle, bb_width = self.calculate_bollinger_bands(prices, window=20)
        signals_df['BB_Upper'] = bb_upper
        signals_df['BB_Lower'] = bb_lower
        signals_df['BB_Middle'] = bb_middle
        signals_df['BB_Width'] = bb_width
        
        # 计算布林带位置
        signals_df['BB_Position'] = (prices - bb_lower) / (bb_upper - bb_lower)
        
        return signals_df
    
    def create_decile_portfolios_benchmark(self, signals_df, returns, strategy_name):
        """
        为基准策略创建十分位投资组合
        
        Args:
            signals_df: 信号DataFrame
            returns: 收益率序列
            strategy_name: 策略名称
            
        Returns:
            portfolio_results: 投资组合结果
        """
        # 获取策略信号
        signal = signals_df[strategy_name].dropna()
        
        # 对齐数据
        common_index = signal.index.intersection(returns.index)
        signal_aligned = signal.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # 创建DataFrame
        df = pd.DataFrame({
            'signal': signal_aligned,
            'return': returns_aligned
        }).dropna()
        
        if len(df) == 0:
            return None
        
        # 按信号排序并分组
        df = df.sort_values('signal').reset_index(drop=True)
        df['decile'] = pd.qcut(df['signal'], q=10, labels=False, duplicates='drop')
        
        # 计算每个十分位的统计量
        decile_stats = df.groupby('decile').agg({
            'return': ['mean', 'std', 'count'],
            'signal': ['mean', 'min', 'max']
        }).round(6)
        
        # 计算夏普比率
        decile_returns = df.groupby('decile')['return'].mean()
        decile_volatilities = df.groupby('decile')['return'].std()
        sharpe_ratios = (decile_returns - self.risk_free_rate/252) / decile_volatilities
        annualized_sharpe = sharpe_ratios * np.sqrt(252)
        
        # 构建结果
        portfolio_results = {
            'strategy_name': strategy_name,
            'decile_stats': decile_stats,
            'decile_returns': decile_returns,
            'decile_volatilities': decile_volatilities,
            'sharpe_ratios': sharpe_ratios,
            'annualized_sharpe': annualized_sharpe,
            'data': df
        }
        
        return portfolio_results
    
    def create_hl_strategy_benchmark(self, portfolio_results):
        """
        为基准策略创建H-L策略
        
        Args:
            portfolio_results: 十分位投资组合结果
            
        Returns:
            hl_results: H-L策略结果
        """
        if portfolio_results is None:
            return None
        
        decile_returns = portfolio_results['decile_returns']
        
        # H-L策略：做多第10分位，做空第1分位
        long_return = decile_returns.iloc[-1]
        short_return = decile_returns.iloc[0]
        hl_return = long_return - short_return
        
        # 计算波动率
        long_vol = portfolio_results['decile_volatilities'].iloc[-1]
        short_vol = portfolio_results['decile_volatilities'].iloc[0]
        hl_volatility = np.sqrt(long_vol**2 + short_vol**2)
        
        # 夏普比率
        hl_sharpe = hl_return / hl_volatility if hl_volatility > 0 else 0
        hl_annualized_sharpe = hl_sharpe * np.sqrt(252)
        
        hl_results = {
            'strategy_name': portfolio_results['strategy_name'],
            'long_return': long_return,
            'short_return': short_return,
            'hl_return': hl_return,
            'hl_volatility': hl_volatility,
            'hl_sharpe': hl_sharpe,
            'hl_annualized_sharpe': hl_annualized_sharpe
        }
        
        return hl_results
    
    def evaluate_all_benchmarks(self, df, returns, price_col='close'):
        """
        评估所有基准策略
        
        Args:
            df: 包含价格数据的DataFrame
            returns: 收益率序列
            price_col: 价格列名
            
        Returns:
            benchmark_results: 所有基准策略结果
        """
        # 创建基准信号
        signals_df = self.create_benchmark_signals(df, price_col)
        
        # 评估每个策略
        strategies = ['MOM', 'STR', 'WSTR', 'TREND', 'RSI', 'BB_Position']
        benchmark_results = {}
        
        for strategy in strategies:
            print(f"评估基准策略: {strategy}")
            
            # 创建十分位投资组合
            portfolio_results = self.create_decile_portfolios_benchmark(
                signals_df, returns, strategy
            )
            
            if portfolio_results is not None:
                # 创建H-L策略
                hl_results = self.create_hl_strategy_benchmark(portfolio_results)
                
                benchmark_results[strategy] = {
                    'portfolio_results': portfolio_results,
                    'hl_results': hl_results
                }
            else:
                print(f"  警告: {strategy} 策略没有有效数据")
                benchmark_results[strategy] = None
        
        return benchmark_results
    
    def compare_with_cnn(self, cnn_results, benchmark_results):
        """
        将CNN结果与基准策略对比
        
        Args:
            cnn_results: CNN模型结果
            benchmark_results: 基准策略结果
            
        Returns:
            comparison_df: 对比结果DataFrame
        """
        comparison_data = []
        
        # 添加CNN结果
        cnn_hl = cnn_results['hl_results']
        comparison_data.append({
            'Strategy': 'CNN',
            'HL_Return': cnn_hl['hl_return'],
            'HL_Volatility': cnn_hl['hl_volatility'],
            'HL_Sharpe': cnn_hl['hl_annualized_sharpe'],
            'Accuracy': cnn_results.get('accuracy', 0),
            'AUC': cnn_results.get('auc', 0)
        })
        
        # 添加基准策略结果
        for strategy_name, results in benchmark_results.items():
            if results is not None and results['hl_results'] is not None:
                hl = results['hl_results']
                comparison_data.append({
                    'Strategy': strategy_name,
                    'HL_Return': hl['hl_return'],
                    'HL_Volatility': hl['hl_volatility'],
                    'HL_Sharpe': hl['hl_annualized_sharpe'],
                    'Accuracy': 0,  # 基准策略没有准确率概念
                    'AUC': 0  # 基准策略没有AUC概念
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('HL_Sharpe', ascending=False)
        
        return comparison_df
    
    def plot_benchmark_comparison(self, comparison_df, save_path=None):
        """
        绘制基准策略对比图
        
        Args:
            comparison_df: 对比结果DataFrame
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. H-L策略夏普比率对比
        axes[0, 0].bar(comparison_df['Strategy'], comparison_df['HL_Sharpe'], 
                      color=['red' if s == 'CNN' else 'skyblue' for s in comparison_df['Strategy']],
                      alpha=0.7)
        axes[0, 0].set_title('H-L策略年化夏普比率对比')
        axes[0, 0].set_ylabel('年化夏普比率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. H-L策略收益率对比
        axes[0, 1].bar(comparison_df['Strategy'], comparison_df['HL_Return'],
                      color=['red' if s == 'CNN' else 'lightgreen' for s in comparison_df['Strategy']],
                      alpha=0.7)
        axes[0, 1].set_title('H-L策略收益率对比')
        axes[0, 1].set_ylabel('收益率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 风险收益散点图
        axes[1, 0].scatter(comparison_df['HL_Volatility'], comparison_df['HL_Return'],
                          s=100, alpha=0.7,
                          c=['red' if s == 'CNN' else 'blue' for s in comparison_df['Strategy']])
        axes[1, 0].set_title('风险收益分析')
        axes[1, 0].set_xlabel('波动率')
        axes[1, 0].set_ylabel('收益率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加策略标签
        for i, strategy in enumerate(comparison_df['Strategy']):
            axes[1, 0].annotate(strategy, 
                               (comparison_df['HL_Volatility'].iloc[i], 
                                comparison_df['HL_Return'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        # 4. 综合性能雷达图（仅CNN）
        cnn_row = comparison_df[comparison_df['Strategy'] == 'CNN']
        if not cnn_row.empty:
            metrics = ['HL_Sharpe', 'HL_Return', 'Accuracy', 'AUC']
            values = [cnn_row[metric].iloc[0] for metric in metrics]
            
            # 归一化到0-1范围
            normalized_values = []
            for i, metric in enumerate(metrics):
                if metric in ['HL_Sharpe', 'HL_Return']:
                    # 对于夏普比率和收益率，使用相对值
                    max_val = comparison_df[metric].max()
                    min_val = comparison_df[metric].min()
                    if max_val != min_val:
                        norm_val = (values[i] - min_val) / (max_val - min_val)
                    else:
                        norm_val = 0.5
                else:
                    # 对于准确率和AUC，直接使用原值
                    norm_val = values[i]
                normalized_values.append(norm_val)
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            normalized_values += normalized_values[:1]
            
            ax_radar = axes[1, 1]
            ax_radar.set_theta_offset(np.pi / 2)
            ax_radar.set_theta_direction(-1)
            ax_radar.plot(angles, normalized_values, 'o-', linewidth=2, color='red', label='CNN')
            ax_radar.fill(angles, normalized_values, alpha=0.25, color='red')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(['夏普比率', '收益率', '准确率', 'AUC'])
            ax_radar.set_title('CNN综合性能')
            ax_radar.legend()
            ax_radar.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"基准策略对比图已保存到: {save_path}")
        
        plt.show()

def main():
    """主函数 - 示例用法"""
    # 这里可以添加示例代码
    pass

if __name__ == '__main__':
    main()
