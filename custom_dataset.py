import torch
from torch.utils.data import Dataset

class TradingDataset(Dataset):
    """自定义交易数据集类，正确处理图像和标签的分离"""
    
    def __init__(self, image_data, label_type='RET5'):
        """
        初始化数据集
        
        Args:
            image_data: 包含[image, ret5, ret20, ret60]的列表
            label_type: 标签类型，'RET5', 'RET20', 或 'RET60'
        """
        self.image_data = image_data
        self.label_type = label_type
        
        # 验证标签类型
        assert label_type in ['RET5', 'RET20', 'RET60'], f"Wrong Label Type: {label_type}"
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            image: 图像数据 (numpy array)
            label: 对应的标签 (int)
        """
        item = self.image_data[idx]
        image, ret5, ret20, ret60 = item
        
        # 根据标签类型选择对应的标签
        if self.label_type == 'RET5':
            label = ret5
        elif self.label_type == 'RET20':
            label = ret20
        else:  # RET60
            label = ret60
        
        # 将图像转换为tensor，并确保正确的维度
        # image shape: (height, width, 1) -> (height, width)
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(2)  # 移除最后一个维度
        
        image_tensor = torch.from_numpy(image).float()
        
        return image_tensor, label
