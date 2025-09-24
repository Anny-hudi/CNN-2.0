from __init__ import *
import utils as _U
reload(_U)
import model as _M
reload(_M)
import train as _T
reload(_T)
import dataset as _D
reload(_D)
import custom_dataset as _CD
reload(_CD)
import multiple_training as _MT
reload(_MT)
import portfolio_evaluation as _PE
reload(_PE)
import benchmark_strategies as _BS
reload(_BS)
import sys


parser = argparse.ArgumentParser(description='Train Models via YAML files')
parser.add_argument('setting', type=str, \
                    help='Experiment Settings, should be yaml files like those in /configs')
parser.add_argument('--multiple_training', action='store_true', \
                    help='Enable multiple training (5 runs) as per paper')
parser.add_argument('--portfolio_evaluation', action='store_true', \
                    help='Enable portfolio evaluation')
parser.add_argument('--benchmark_comparison', action='store_true', \
                    help='Enable benchmark strategy comparison')

args = parser.parse_args()

with open(args.setting, 'r') as f:
    setting = _U.Dict2ObjParser(yaml.safe_load(f)).parse()

# 创建必要的目录
if 'models' not in os.listdir('./'):
    os.system('mkdir models')
if setting.TRAIN.MODEL_SAVE_FILE.split('/')[1] not in os.listdir('./models/'):
    os.system(f"cd models && mkdir {setting.TRAIN.MODEL_SAVE_FILE.split('/')[1]}")
if 'logs' not in os.listdir('./'):
    os.system('mkdir logs')
if setting.TRAIN.LOG_SAVE_FILE.split('/')[1] not in os.listdir('./logs/'):
    os.system(f"cd logs && mkdir {setting.TRAIN.LOG_SAVE_FILE.split('/')[1]}")

# 检查模型是否已存在
dir = setting.TRAIN.MODEL_SAVE_FILE.split('/')[0] + '/' + setting.TRAIN.MODEL_SAVE_FILE.split('/')[1]
if setting.TRAIN.MODEL_SAVE_FILE.split('/')[2] in os.listdir(dir):
    print(f'Pretrained Model: {args.setting} Already Exist')
    if not args.multiple_training:
        sys.exit(0)

if __name__ == '__main__':
    
    if args.multiple_training:
        print("="*60)
        print("执行多次训练（符合论文要求）")
        print("="*60)
        
        # 使用多次训练管理器
        trainer = _MT.MultipleTrainingManager(setting, n_training_runs=5)
        ensemble_model, training_results = trainer.train_multiple_runs()
        
        print(f"\n多次训练完成！集成模型已保存。")
        
        # 如果启用投资组合评估
        if args.portfolio_evaluation:
            print("\n开始投资组合评估...")
            
            # 准备测试数据
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
            
            test_images = test_dataset.generate_images(1.0)
            
            if len(test_images) > 0:
                # 准备测试数据
                test_data = torch.stack([torch.FloatTensor(img[0]) for img in test_images])
                test_labels = torch.LongTensor([img[1] for img in test_images] if setting.TRAIN.LABEL == 'RET5' 
                                             else [img[2] for img in test_images])
                
                # 投资组合评估
                evaluator = _PE.PortfolioEvaluator()
                evaluation_results = evaluator.evaluate_model_performance(
                    ensemble_model, test_data, test_labels
                )
                
                # 生成报告
                model_name = f"{setting.MODEL}_{setting.TRAIN.LABEL}"
                evaluator.generate_portfolio_report(
                    evaluation_results, 
                    model_name,
                    save_path=f"results/{model_name}_portfolio_report.csv"
                )
                
                # 绘制分析图
                evaluator.plot_decile_analysis(
                    evaluation_results['portfolio_results'],
                    save_path=f"results/{model_name}_decile_analysis.png"
                )
                
                evaluator.plot_hl_strategy_analysis(
                    evaluation_results['hl_results'],
                    save_path=f"results/{model_name}_hl_strategy.png"
                )
                
                # 如果启用基准策略对比
                if args.benchmark_comparison:
                    print("\n开始基准策略对比...")
                    
                    # 准备基准策略数据
                    benchmark_evaluator = _BS.BenchmarkStrategies()
                    
                    # 这里需要从测试数据中提取价格序列
                    # 简化版本：使用模拟数据
                    print("注意：基准策略对比需要完整的价格序列数据")
                    print("当前版本使用模拟数据进行演示")
                    
                    # 创建模拟价格数据
                    np.random.seed(42)
                    n_days = 1000
                    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
                    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, n_days))
                    returns = np.diff(prices) / prices[:-1]
                    
                    df = pd.DataFrame({
                        'date': dates,
                        'close': prices
                    })
                    
                    # 评估基准策略
                    benchmark_results = benchmark_evaluator.evaluate_all_benchmarks(
                        df, pd.Series(returns, index=dates[1:])
                    )
                    
                    # 对比CNN与基准策略
                    comparison_df = benchmark_evaluator.compare_with_cnn(
                        evaluation_results, benchmark_results
                    )
                    
                    # 保存对比结果
                    comparison_df.to_csv(f"results/{model_name}_benchmark_comparison.csv", index=False)
                    
                    # 绘制对比图
                    benchmark_evaluator.plot_benchmark_comparison(
                        comparison_df,
                        save_path=f"results/{model_name}_benchmark_comparison.png"
                    )
                    
                    print(f"基准策略对比完成！结果已保存到 results/{model_name}_benchmark_comparison.csv")
                
                print(f"投资组合评估完成！结果已保存到 results/ 目录")
            else:
                print("警告：没有生成测试数据，跳过投资组合评估")
        
    else:
        print("="*60)
        print("执行单次训练（原始方法）")
        print("="*60)
        
        # 原始单次训练逻辑
        dataset = _D.ImageDataSet(win_size = setting.DATASET.LOOKBACK_WIN, \
                                    start_date = setting.DATASET.START_DATE, \
                                    end_date = setting.DATASET.END_DATE, \
                                    mode = 'train', \
                                    label = setting.TRAIN.LABEL, \
                                    indicators = setting.DATASET.INDICATORS, \
                                    show_volume = setting.DATASET.SHOW_VOLUME, \
                                    parallel_num=setting.DATASET.PARALLEL_NUM)

        image_set = dataset.generate_images(setting.DATASET.SAMPLE_RATE)

        # 使用自定义数据集类来正确处理图像和标签
        trading_dataset = _CD.TradingDataset(image_set, setting.TRAIN.LABEL)

        train_loader_size = int(len(trading_dataset)*(1-setting.TRAIN.VALID_RATIO))
        valid_loader_size = len(trading_dataset) - train_loader_size

        train_dataset, valid_dataset = torch.utils.data.random_split(trading_dataset, [train_loader_size, valid_loader_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert setting.MODEL in ['CNN5d', 'CNN20d', 'CNN60d'], f"Wrong Model Template: {setting.MODEL}"

        if setting.MODEL == 'CNN5d':
            model = _M.CNN5d()
        elif setting.MODEL == 'CNN20d':
            model = _M.CNN20d()
        else:
            model = _M.CNN60d()
        model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=setting.TRAIN.LEARNING_RATE, weight_decay=setting.TRAIN.WEIGHT_DECAY)
        
        train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = _T.train_n_epochs(setting.TRAIN.NEPOCH, model, setting.TRAIN.LABEL, train_loader, valid_loader, criterion, optimizer, setting.TRAIN.MODEL_SAVE_FILE, setting.TRAIN.EARLY_STOP_EPOCH)
        
        log = pd.DataFrame([train_loss_set, train_acc_set, valid_loss_set, valid_acc_set], index=['train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
        log.to_csv(setting.TRAIN.LOG_SAVE_FILE)
        
        print("单次训练完成！")
    
    