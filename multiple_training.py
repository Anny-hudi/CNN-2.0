"""
多次训练模块 - 按照论文要求每个模型训练5次取平均
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
from train import train_n_epochs
import model as _M
import dataset as _D
import custom_dataset as _CD
import utils as _U

class MultipleTrainingManager:
    """多次训练管理器"""
    
    def __init__(self, setting, n_training_runs=5):
        """
        初始化多次训练管理器
        
        Args:
            setting: 训练配置
            n_training_runs: 训练次数，默认5次
        """
        self.setting = setting
        self.n_training_runs = n_training_runs
        self.training_results = []
        self.best_models = []
        
    def prepare_data(self):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 创建数据集
        dataset = _D.ImageDataSet(
            win_size=self.setting.DATASET.LOOKBACK_WIN,
            start_date=self.setting.DATASET.START_DATE,
            end_date=self.setting.DATASET.END_DATE,
            mode='train',
            label=self.setting.TRAIN.LABEL,
            indicators=self.setting.DATASET.INDICATORS,
            show_volume=self.setting.DATASET.SHOW_VOLUME,
            parallel_num=self.setting.DATASET.PARALLEL_NUM
        )
        
        # 生成图像数据
        image_set = dataset.generate_images(self.setting.DATASET.SAMPLE_RATE)
        
        # 使用自定义数据集类
        trading_dataset = _CD.TradingDataset(image_set, self.setting.TRAIN.LABEL)
        
        # 分割训练和验证集
        train_loader_size = int(len(trading_dataset) * (1 - self.setting.TRAIN.VALID_RATIO))
        valid_loader_size = len(trading_dataset) - train_loader_size
        
        train_dataset, valid_dataset = torch.utils.data.random_split(
            trading_dataset, [train_loader_size, valid_loader_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=self.setting.TRAIN.BATCH_SIZE, 
            shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, 
            batch_size=self.setting.TRAIN.BATCH_SIZE, 
            shuffle=True
        )
        
        return train_loader, valid_loader
    
    def create_model(self):
        """创建模型"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.setting.MODEL == 'CNN5d':
            model = _M.CNN5d()
        elif self.setting.MODEL == 'CNN20d':
            model = _M.CNN20d()
        elif self.setting.MODEL == 'CNN60d':
            model = _M.CNN60d()
        else:
            raise ValueError(f"不支持的模型类型: {self.setting.MODEL}")
        
        model.to(device)
        return model, device
    
    def train_single_run(self, run_id, train_loader, valid_loader):
        """单次训练"""
        print(f"\n{'='*50}")
        print(f"开始第 {run_id + 1}/{self.n_training_runs} 次训练")
        print(f"{'='*50}")
        
        # 创建模型
        model, device = self.create_model()
        
        # 设置优化器和损失函数
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.setting.TRAIN.LEARNING_RATE, 
            weight_decay=self.setting.TRAIN.WEIGHT_DECAY
        )
        
        # 生成唯一的保存路径
        base_save_path = self.setting.TRAIN.MODEL_SAVE_FILE
        base_log_path = self.setting.TRAIN.LOG_SAVE_FILE
        
        # 为每次训练创建唯一的文件名
        save_dir = os.path.dirname(base_save_path)
        log_dir = os.path.dirname(base_log_path)
        
        save_filename = os.path.basename(base_save_path).replace('.tar', f'_run{run_id+1}.tar')
        log_filename = os.path.basename(base_log_path).replace('.csv', f'_run{run_id+1}.csv')
        
        run_save_path = os.path.join(save_dir, save_filename)
        run_log_path = os.path.join(log_dir, log_filename)
        
        # 训练模型
        train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = train_n_epochs(
            self.setting.TRAIN.NEPOCH,
            model,
            self.setting.TRAIN.LABEL,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            run_save_path,
            self.setting.TRAIN.EARLY_STOP_EPOCH
        )
        
        # 保存训练日志
        import pandas as pd
        log = pd.DataFrame([train_loss_set, train_acc_set, valid_loss_set, valid_acc_set], 
                          index=['train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
        log.to_csv(run_log_path)
        
        # 加载最佳模型
        checkpoint = torch.load(run_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 记录训练结果
        result = {
            'run_id': run_id + 1,
            'model': model,
            'train_loss': train_loss_set,
            'valid_loss': valid_loss_set,
            'train_acc': train_acc_set,
            'valid_acc': valid_acc_set,
            'best_epoch': checkpoint['epoch'],
            'save_path': run_save_path,
            'log_path': run_log_path
        }
        
        self.training_results.append(result)
        self.best_models.append(copy.deepcopy(model))
        
        print(f"第 {run_id + 1} 次训练完成")
        print(f"最佳验证损失: {min(valid_loss_set):.6f}")
        print(f"最佳验证准确率: {max(valid_acc_set):.6f}")
        
        return result
    
    def create_ensemble_model(self):
        """创建集成模型（平均权重）"""
        if len(self.best_models) == 0:
            raise ValueError("没有训练好的模型")
        
        print(f"\n创建集成模型，平均 {len(self.best_models)} 个模型的权重...")
        
        # 获取第一个模型作为基础
        ensemble_model = copy.deepcopy(self.best_models[0])
        
        # 平均所有权重
        for param in ensemble_model.parameters():
            param.data.zero_()
        
        for model in self.best_models:
            for param, model_param in zip(ensemble_model.parameters(), model.parameters()):
                param.data += model_param.data
        
        # 除以模型数量得到平均值
        for param in ensemble_model.parameters():
            param.data /= len(self.best_models)
        
        return ensemble_model
    
    def save_ensemble_model(self, ensemble_model):
        """保存集成模型"""
        ensemble_save_path = self.setting.TRAIN.MODEL_SAVE_FILE.replace('.tar', '_ensemble.tar')
        
        # 只保存模型状态字典，不保存配置对象
        torch.save({
            'model_state_dict': ensemble_model.state_dict(),
            'n_models': len(self.best_models),
            'training_runs': self.n_training_runs,
            'model_type': self.setting.MODEL,
            'label_type': self.setting.TRAIN.LABEL
        }, ensemble_save_path)
        
        print(f"集成模型已保存到: {ensemble_save_path}")
        return ensemble_save_path
    
    def train_multiple_runs(self):
        """执行多次训练"""
        print(f"开始多次训练，共 {self.n_training_runs} 次...")
        
        # 准备数据
        train_loader, valid_loader = self.prepare_data()
        
        # 执行多次训练
        for run_id in range(self.n_training_runs):
            try:
                result = self.train_single_run(run_id, train_loader, valid_loader)
            except Exception as e:
                print(f"第 {run_id + 1} 次训练失败: {e}")
                continue
        
        if len(self.training_results) == 0:
            raise ValueError("所有训练都失败了")
        
        # 创建集成模型
        ensemble_model = self.create_ensemble_model()
        
        # 保存集成模型
        ensemble_path = self.save_ensemble_model(ensemble_model)
        
        # 生成训练报告
        self.generate_training_report()
        
        return ensemble_model, self.training_results
    
    def generate_training_report(self):
        """生成训练报告"""
        print(f"\n{'='*60}")
        print("多次训练报告")
        print(f"{'='*60}")
        
        print(f"训练次数: {self.n_training_runs}")
        print(f"成功训练: {len(self.training_results)} 次")
        
        # 统计最佳性能
        best_valid_losses = [min(result['valid_loss']) for result in self.training_results]
        best_valid_accs = [max(result['valid_acc']) for result in self.training_results]
        
        print(f"\n验证损失统计:")
        print(f"  平均: {np.mean(best_valid_losses):.6f}")
        print(f"  标准差: {np.std(best_valid_losses):.6f}")
        print(f"  最小值: {np.min(best_valid_losses):.6f}")
        print(f"  最大值: {np.max(best_valid_losses):.6f}")
        
        print(f"\n验证准确率统计:")
        print(f"  平均: {np.mean(best_valid_accs):.6f}")
        print(f"  标准差: {np.std(best_valid_accs):.6f}")
        print(f"  最小值: {np.min(best_valid_accs):.6f}")
        print(f"  最大值: {np.max(best_valid_accs):.6f}")
        
        # 保存详细报告
        import pandas as pd
        report_data = []
        for result in self.training_results:
            report_data.append({
                'run_id': result['run_id'],
                'best_epoch': result['best_epoch'],
                'best_valid_loss': min(result['valid_loss']),
                'best_valid_acc': max(result['valid_acc']),
                'final_train_loss': result['train_loss'][-1],
                'final_train_acc': result['train_acc'][-1]
            })
        
        report_df = pd.DataFrame(report_data)
        report_path = self.setting.TRAIN.LOG_SAVE_FILE.replace('.csv', '_multiple_training_report.csv')
        report_df.to_csv(report_path, index=False)
        print(f"\n详细报告已保存到: {report_path}")

def main():
    """主函数 - 示例用法"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='多次训练CNN模型')
    parser.add_argument('setting', type=str, help='配置文件路径')
    parser.add_argument('--runs', type=int, default=5, help='训练次数，默认5次')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.setting, 'r') as f:
        setting = _U.Dict2ObjParser(yaml.safe_load(f)).parse()
    
    # 创建多次训练管理器
    trainer = MultipleTrainingManager(setting, args.runs)
    
    # 执行多次训练
    ensemble_model, results = trainer.train_multiple_runs()
    
    print("多次训练完成！")

if __name__ == '__main__':
    main()
