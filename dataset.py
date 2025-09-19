from __init__ import *
import utils as _U
reload(_U)
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator

SUPPORTED_INDICATORS = ['MA']

def cal_indicators(tabular_df, indicator_name, parameters):
    if indicator_name == "MA":
        assert len(parameters) == 1, f'Wrong parameters num, expected 1, got {len(parameters)}'
        slice_win_size = int(parameters[0])
        MA = tabular_df['close'].rolling(slice_win_size, min_periods=1).mean()
        return MA # pd.Series



def single_symbol_image(tabular_df, image_size, start_date, sample_rate, indicators, show_volume, mode):
    ''' generate Candlelist images
    
    parameters: [
        tabular_df  -> pandas.DataFrame: tabular data,
        image_size  -> tuple: (H, W), size shouble (32, 15), (64, 60)
        start_date  -> int: truncate extra rows after generating images,
        indicators  -> dict: technical indicators added on the image, e.g. {"MA": [20]},
        show_volume -> boolean: show volume bars or not
        mode        -> 'train': for train & validation; 'test': for test; 'inference': for inference
    ]
    
    Note: A single day's data occupies 3 pixel (width). First rows's dates should be prior to the start date in order to make sure there are enough data to generate image for the start date.
    
    return -> list: each item of the list is [np.array(image_size), binary, binary, binary]. The last two binary (0./1.) are the label of ret5, ret20
    
    '''
    
    
    ind_names = []
    if indicators:
        for i in range(len(indicators)//2):
            ind = indicators[i*2].NAME
            ind_names.append(ind)
            params = str(indicators[i*2+1].PARAM).split(' ')
            tabular_df[ind] = cal_indicators(tabular_df, ind, params)
    
    dataset = []
    valid_dates = []
    lookback = image_size[1]//3
    for d in range(lookback-1, len(tabular_df)):
        # random skip some trading dates
        if np.random.rand(1) > sample_rate:
            continue
        # skip dates before start_date
        if tabular_df.iloc[d]['date'] < start_date:
            continue
        
        price_slice = tabular_df[d-(lookback-1):d+1][['open', 'high', 'low', 'close']+ind_names].reset_index(drop=True)
        volume_slice = tabular_df[d-(lookback-1):d+1][['volume']].reset_index(drop=True)

        # number of no transactions days > 0.2*look back days
        if (1.0*(price_slice[['open', 'high', 'low', 'close']].sum(axis=1)/price_slice['open'] == 4)).sum() > lookback//5: 
            continue
        
        valid_dates.append(tabular_df.iloc[d]['date']) # trading dates surviving the validation
        
        # project price into quantile
        price_slice = (price_slice - np.min(price_slice.values))/(np.max(price_slice.values) - np.min(price_slice.values))
        volume_slice = (volume_slice - np.min(volume_slice.values))/(np.max(volume_slice.values) - np.min(volume_slice.values))

        if not show_volume:
            price_slice = price_slice.apply(lambda x: x*(image_size[0]-1)).astype(int)
        else:
            if image_size[0] == 32:
                price_slice = price_slice.apply(lambda x: x*(25-1)+7).astype(int)
                volume_slice = volume_slice.apply(lambda x: x*(6-1)).astype(int)
            else:
                price_slice = price_slice.apply(lambda x: x*(51-1)+13).astype(int)
                volume_slice = volume_slice.apply(lambda x: x*(12-1)).astype(int)
        
        image = np.zeros(image_size)
        for i in range(len(price_slice)):
            # draw candlelist 
            image[price_slice.loc[i]['open'], i*3] = 255.
            image[price_slice.loc[i]['low']:price_slice.loc[i]['high']+1, i*3+1] = 255.
            image[price_slice.loc[i]['close'], i*3+2] = 255.
            # draw indicators
            for ind in ind_names:
                image[price_slice.loc[i][ind], i*3:i*3+2] = 255.
            # draw volume bars
            if show_volume:
                image[:volume_slice.loc[i]['volume'], i*3+1] = 255.
    
        label_ret5 = 1 if np.sign(tabular_df.iloc[d]['ret5']) > 0 else 0
        label_ret20 = 1 if np.sign(tabular_df.iloc[d]['ret20']) > 0 else 0
        label_ret60 = 1 if np.sign(tabular_df.iloc[d]['ret60']) > 0 else 0
        
        entry = [image, label_ret5, label_ret20, label_ret60]
        dataset.append(entry)
    
    if mode == 'train' or mode == 'test':
        return dataset
    else:
        return [tabular_df.iloc[0]['code'], dataset, valid_dates]


class ImageDataSet():
    def __init__(self, win_size, start_date, end_date, mode, label, indicators=[], show_volume=False, parallel_num=-1):
        ## Check whether inputs are valid
        assert isinstance(start_date, int) and isinstance(end_date, int), f'Type Error: start_date & end_date shoule be int'
        assert start_date < end_date, f'start date {start_date} cannnot be later than end date {end_date}'
        assert win_size in [5, 20, 60], f'Wrong look back days: {win_size}'
        assert mode in ['train', 'test', 'inference'], f'Type Error: {mode}'
        assert label in ['RET5', 'RET20', 'RET60'], f'Wrong Label: {label}'
        assert indicators is None or len(indicators)%2 == 0, 'Config Error, length of indicators should be even'
        if indicators:
            for i in range(len(indicators)//2):
                assert indicators[2*i].NAME in SUPPORTED_INDICATORS, f"Error: Calculation of {indicators[2*i].NAME} is not defined"
        
        ## Attributes of ImageDataSet
        if win_size == 5:
            self.image_size = (32, 15)
            self.extra_dates = datetime.timedelta(days=40)
        elif win_size == 20:
            self.image_size = (64, 60)
            self.extra_dates = datetime.timedelta(days=40)
        elif win_size == 60:
            self.image_size = (96, 180)
            self.extra_dates = datetime.timedelta(days=40)
            
        self.start_date = start_date
        self.end_date = end_date 
        self.mode = mode
        self.label = label
        self.indicators = indicators
        self.show_volume = show_volume
        self.parallel_num = parallel_num
        self.win_size = win_size
        
        # 初始化数据处理器和图像生成器
        self.data_processor = StockDataProcessor()
        self.image_generator = OHLCImageGenerator(win_size)
        
        ## Load data from zipfile
        self.load_data()
        
        # Log info
        if indicators:
            ind_info = [(self.indicators[2*i].NAME, str(self.indicators[2*i+1].PARAM).split(' ')) for i in range(len(self.indicators)//2)]
        else:
            ind_info = []
        print(f"DataSet Initialized\n \t - Mode:         {self.mode.upper()}\n \t - Image Size:   {self.image_size}\n \t - Time Period:  {self.start_date} - {self.end_date}\n \t - Indicators:   {ind_info}\n \t - Volume Shown: {self.show_volume}")
        
        
    @_U.timer('Load Data', '8')
    def load_data(self):
        # 使用新的数据处理器加载数据
        self.data_processor.load_data()
        
        # 为了保持兼容性，仍然创建self.df
        # 但这里我们使用新的数据结构
        all_data = []
        for code, df in self.data_processor.data.items():
            # 转换日期格式以保持兼容性
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y%m%d').astype(int)
            df_copy['code'] = code
            all_data.append(df_copy)
        
        if all_data:
            self.df = pd.concat(all_data, axis=0, ignore_index=True)
            
            # 计算收益率（保持原有逻辑）
            self.df['ret5'] = np.zeros(self.df.shape[0])
            self.df['ret20'] = np.zeros(self.df.shape[0])
            self.df['ret60'] = np.zeros(self.df.shape[0])
            self.df = self.df.sort_values(['code', 'date']).reset_index(drop=True)
            self.df['ret5'] = self.df.groupby('code')['close'].pct_change(5) * 100
            self.df['ret20'] = self.df.groupby('code')['close'].pct_change(20) * 100
            self.df['ret60'] = self.df.groupby('code')['close'].pct_change(60) * 100
            self.df['ret5'] = self.df.groupby('code')['ret5'].shift(-5)
            self.df['ret20'] = self.df.groupby('code')['ret20'].shift(-20)
            self.df['ret60'] = self.df.groupby('code')['ret60'].shift(-60)

            # 过滤日期范围
            self.df = self.df.loc[self.df['date'] <= self.end_date]
        else:
            raise ValueError("未能加载任何数据")
        
        
    def generate_images(self, sample_rate):
        """使用新的图像生成逻辑生成图像"""
        # 获取所有股票代码
        symbols = list(self.data_processor.data.keys())
        
        all_sequences = []
        all_labels = []
        all_dates = []
        
        # 为每个股票生成序列数据
        for symbol in symbols:
            try:
                sequences, labels, dates = self.data_processor.get_processed_data(
                    symbol, self.win_size, self.win_size
                )
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                all_dates.extend(dates)
            except Exception as e:
                print(f"处理股票 {symbol} 时出错: {e}")
                continue
        
        if len(all_sequences) == 0:
            print("警告：没有生成任何序列数据")
            return []
        
        # 根据标签类型选择对应的标签
        if self.label == 'RET5':
            # 对于RET5，我们需要重新计算5天收益率
            target_labels = []
            for i, seq in enumerate(all_sequences):
                if i + 5 < len(all_sequences):
                    current_price = seq['Adj_Close_calc'].iloc[-1]
                    future_price = all_sequences[i + 5]['Adj_Close_calc'].iloc[-1]
                    future_return = (future_price - current_price) / current_price
                    target_labels.append(int(1 if future_return > 0 else 0))
                else:
                    target_labels.append(0)  # 默认标签
        elif self.label == 'RET20':
            # 对于RET20，我们需要重新计算20天收益率
            target_labels = []
            for i, seq in enumerate(all_sequences):
                if i + 20 < len(all_sequences):
                    current_price = seq['Adj_Close_calc'].iloc[-1]
                    future_price = all_sequences[i + 20]['Adj_Close_calc'].iloc[-1]
                    future_return = (future_price - current_price) / current_price
                    target_labels.append(int(1 if future_return > 0 else 0))
                else:
                    target_labels.append(0)  # 默认标签
        elif self.label == 'RET60':
            # 对于RET60，我们需要重新计算60天收益率
            target_labels = []
            for i, seq in enumerate(all_sequences):
                if i + 60 < len(all_sequences):
                    current_price = seq['Adj_Close_calc'].iloc[-1]
                    future_price = all_sequences[i + 60]['Adj_Close_calc'].iloc[-1]
                    future_return = (future_price - current_price) / current_price
                    target_labels.append(int(1 if future_return > 0 else 0))
                else:
                    target_labels.append(0)  # 默认标签
        else:
            target_labels = all_labels
        
        # 应用采样率
        if sample_rate < 1.0:
            import random
            random.seed(42)  # 固定随机种子以确保可重复性
            sampled_indices = random.sample(range(len(all_sequences)), 
                                          int(len(all_sequences) * sample_rate))
            all_sequences = [all_sequences[i] for i in sampled_indices]
            target_labels = [target_labels[i] for i in sampled_indices]
            all_dates = [all_dates[i] for i in sampled_indices]
        
        # 先计算训练集统计量（如果还没有计算的话）
        if self.image_generator.train_mean is None or self.image_generator.train_std is None:
            print("计算训练集统计量...")
            # 使用前70%的数据作为训练集来计算统计量
            train_size = int(len(all_sequences) * 0.7)
            if train_size > 0:
                train_sequences = all_sequences[:train_size]
                self.image_generator.fit_normalizer(train_sequences)
            else:
                print("警告：训练集太小，跳过统计量计算")
        
        # 使用新的图像生成器生成图像
        images = self.image_generator.generate_batch(all_sequences)
        
        # 构建数据集
        dataset = []
        for i in range(len(images)):
            # 为了保持兼容性，我们仍然返回[image, ret5, ret20, ret60]格式
            # 但这里我们使用计算出的目标标签
            if self.label == 'RET5':
                ret5 = target_labels[i]
                ret20 = 0  # 占位符
                ret60 = 0  # 占位符
            elif self.label == 'RET20':
                ret5 = 0   # 占位符
                ret20 = target_labels[i]
                ret60 = 0  # 占位符
            elif self.label == 'RET60':
                ret5 = 0   # 占位符
                ret20 = 0  # 占位符
                ret60 = target_labels[i]
            else:
                ret5 = ret20 = ret60 = 0
            
            dataset.append([images[i], ret5, ret20, ret60])
        
        if self.mode == 'train' and len(dataset) > 0:
            # 处理类别不平衡（保持原有逻辑）
            image_set = pd.DataFrame(dataset, columns=['img', 'ret5', 'ret20', 'ret60'])
            image_set['index'] = image_set.index
            
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE()
                
                if self.label == 'RET5':
                    num0_before = image_set.loc[image_set['ret5'] == 0].shape[0]
                    num1_before = image_set.loc[image_set['ret5'] == 1].shape[0]
                    if num0_before > 0 and num1_before > 0:
                        resample_index, _ = smote.fit_resample(image_set[['index', 'ret20', 'ret60']], image_set['ret5'])
                        image_set = image_set[['img', 'ret5', 'ret20', 'ret60']].loc[resample_index['index']]
                    num0 = image_set.loc[image_set['ret5'] == 0].shape[0]
                    num1 = image_set.loc[image_set['ret5'] == 1].shape[0]
                    image_set = image_set.values.tolist()
                    
                elif self.label == 'RET20':
                    num0_before = image_set.loc[image_set['ret20'] == 0].shape[0]
                    num1_before = image_set.loc[image_set['ret20'] == 1].shape[0]
                    if num0_before > 0 and num1_before > 0:
                        resample_index, _ = smote.fit_resample(image_set[['index', 'ret5', 'ret60']], image_set['ret20'])
                        image_set = image_set[['img', 'ret5', 'ret20', 'ret60']].loc[resample_index['index']]
                    num0 = image_set.loc[image_set['ret20'] == 0].shape[0]
                    num1 = image_set.loc[image_set['ret20'] == 1].shape[0]
                    image_set = image_set.values.tolist()
                    
                else:  # RET60
                    num0_before = image_set.loc[image_set['ret60'] == 0].shape[0]
                    num1_before = image_set.loc[image_set['ret60'] == 1].shape[0]
                    if num0_before > 0 and num1_before > 0:
                        resample_index, _ = smote.fit_resample(image_set[['index', 'ret5', 'ret20']], image_set['ret60'])
                        image_set = image_set[['img', 'ret5', 'ret20', 'ret60']].loc[resample_index['index']]
                    num0 = image_set.loc[image_set['ret60'] == 0].shape[0]
                    num1 = image_set.loc[image_set['ret60'] == 1].shape[0]
                    image_set = image_set.values.tolist()
                    
                print(f"LABEL: {self.label}\n\tBefore Resample: 0: {num0_before}/{num0_before+num1_before}, 1: {num1_before}/{num0_before+num1_before}\n\tResampled ImageSet: 0: {num0}/{num0+num1}, 1: {num1}/{num0+num1}")
                
            except ImportError:
                print("警告：未安装imbalanced-learn，跳过SMOTE重采样")
                image_set = image_set.values.tolist()
        else:
            image_set = dataset
        
        return image_set