"""
数据加载与预处理模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

class StockDataProcessor:
    def __init__(self, data_fraction=1.0):
        """
        初始化数据处理器
        
        Args:
            data_fraction: 使用的数据比例 (0.0-1.0)，默认1.0使用全部数据
                          设置为0.1则只使用10%的数据（测试用）
        """
        self.data = {}
        self.data_fraction = data_fraction
        
    def load_data(self):
        """加载所有股票数据"""
        data_dir = Path("data")
        
        if not data_dir.exists():
            raise FileNotFoundError("未找到数据目录 data/。请将 CSV 文件放置于项目根目录下的 data/。")
        
        csv_files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
        if len(csv_files) == 0:
            raise FileNotFoundError("data/ 目录下未发现 CSV 文件。")
        
        all_frames = []
        for fn in csv_files:
            full_path = data_dir / fn
            try:
                df = pd.read_csv(full_path)
            except Exception as e:
                print(f"读取失败: {full_path} -> {e}")
                continue

            # 标准化列名
            df.columns = [c.strip().lower() for c in df.columns]
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in df.columns]
            if len(missing) > 0:
                print(f"{fn} 缺少必要列: {missing}")
                continue

            # 解析和排序日期
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 添加股票代码
            if '^GSPC' in fn:
                code = '^GSPC'
            elif '^IXIC' in fn:
                code = '^IXIC'
            elif '^DJI' in fn:
                code = '^DJI'
            else:
                code = os.path.splitext(fn)[0]
            df['code'] = code

            # 只保留必要的列
            df = df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
            all_frames.append(df)

        if len(all_frames) == 0:
            raise ValueError("未能从 data/ 读取到任何有效数据帧。")
        
        tabularDf = pd.concat(all_frames, axis=0, ignore_index=True)

        # 转换日期为整数格式 YYYYMMDD
        tabularDf['date'] = tabularDf['date'].dt.strftime('%Y%m%d').astype(int)

        # 按股票代码分组处理
        for code, group in tabularDf.groupby('code'):
            if self.data_fraction < 1.0:
                total_rows = len(group)
                use_rows = int(total_rows * self.data_fraction)
                print(f"数据量控制：{code} 原始 {total_rows} 行 → 使用 {use_rows} 行 ({self.data_fraction*100:.1f}%)")
                group = group.head(use_rows)
            
            self.data[code] = self._clean_data(group)
        
        return self.data
    
    def _clean_data(self, df):
        """数据清洗和预处理"""
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 计算日收益率
        df['Return'] = df['close'].pct_change()
        
        # 标记缺失数据类型
        df['has_ohlc'] = ~(df['open'].isna() | df['high'].isna() | 
                          df['low'].isna() | df['close'].isna())
        df['has_volume'] = ~df['volume'].isna()
        df['has_high_low'] = ~(df['high'].isna() | df['low'].isna())
        
        # 处理异常值：移除极端的价格变化
        df['price_change'] = df['close'].pct_change()
        df = df[abs(df['price_change']) < 0.5]  # 移除单日涨跌幅超过50%的异常值
        
        # 重新计算收益率
        df['Return'] = df['close'].pct_change()
        
        return df
    
    def adjust_prices(self, df):
        """按照论文要求进行价格标准化处理"""
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 计算日收益率
        df['Return'] = df['close'].pct_change()
        
        # 按照论文要求：首日收盘价设为1，基于回报率计算后续价格
        df['normalized_close'] = 1.0  # 首日收盘价设为1
        
        # 基于回报率计算后续每日收盘价: pt+1 = (1 + RETt+1) * pt
        for i in range(1, len(df)):
            if not pd.isna(df.loc[i, 'Return']):
                df.loc[i, 'normalized_close'] = (1 + df.loc[i, 'Return']) * df.loc[i-1, 'normalized_close']
            else:
                df.loc[i, 'normalized_close'] = df.loc[i-1, 'normalized_close']
        
        # 按当天收盘价水平的比例来划分开盘价、最高价和最低价
        # 计算原始价格相对于收盘价的比例
        df['open_ratio'] = df['open'] / df['close']
        df['high_ratio'] = df['high'] / df['close']
        df['low_ratio'] = df['low'] / df['close']
        
        # 应用比例到标准化后的收盘价
        df['Adj_Open'] = df['open_ratio'] * df['normalized_close']
        df['Adj_High'] = df['high_ratio'] * df['normalized_close']
        df['Adj_Low'] = df['low_ratio'] * df['normalized_close']
        df['Adj_Close_calc'] = df['normalized_close']
        
        # 确保收盘价完全匹配标准化后的价格
        df['Adj_Close_calc'] = df['normalized_close']
        
        # 清理临时列
        df = df.drop(['open_ratio', 'high_ratio', 'low_ratio'], axis=1)
        
        return df
    
    def create_sequences(self, df, window_days, prediction_days):
        """创建时间序列和标签
        
        Args:
            window_days: 图像窗口长度（监督期）
            prediction_days: 预测持有期（应与window_days相同）
        
        Returns:
            sequences: 窗口期数据列表
            labels: 标签列表
            dates: 每个序列对应的最后一天日期
        """
        sequences = []
        labels = []
        dates = []
        
        # 监督期=持有期
        if window_days != prediction_days:
            print(f"警告：监督期({window_days})与持有期({prediction_days})不一致，调整为相同")
            prediction_days = window_days
        
        for i in range(len(df) - window_days - prediction_days + 1):
            # 获取窗口期数据
            window_data = df.iloc[i:i+window_days].copy()
            
            # 检查当前价格和未来价格是否有效
            current_price = df.iloc[i+window_days-1]['Adj_Close_calc']
            future_price = df.iloc[i+window_days+prediction_days-1]['Adj_Close_calc']
            
            # 跳过价格数据无效的序列
            if pd.isna(current_price) or pd.isna(future_price):
                continue
                
            # 计算未来持有期累计收益
            future_return = (future_price - current_price) / current_price
            
            # 二分类标签（未来持有期累计回报是否为正）
            sequences.append(window_data)
            labels.append(int(1 if future_return > 0 else 0))
            dates.append(pd.to_datetime(str(df.iloc[i+window_days-1]['date'])))  # 添加序列最后一天的日期
            
        return sequences, labels, dates
    
    def get_processed_data(self, symbol, window_days, prediction_days):
        """获取处理后的数据"""
        if symbol not in self.data:
            raise ValueError(f"Symbol {symbol} not found")
            
        df = self.data[symbol].copy()
        df = self.adjust_prices(df)
        
        sequences, labels, dates = self.create_sequences(df, window_days, prediction_days)
        
        return sequences, labels, dates
