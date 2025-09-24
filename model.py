# each day is displayed in 3 pixels
# use cnn to predict 
from __init__ import *


class CNN5d(nn.Module):
    # Input: [N, (1), 32, 15]; Output: [N, 2]
    # Two Convolution Blocks
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN5d, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 64, 32, 15]
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=(2,1))) # output size: [N, 64, 16, 15]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 128, 16, 15]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=(2,1))) # output size: [N, 128, 8, 15]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)

        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(15360, 2)
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x): # input: [N, 32, 15]
        x = x.unsqueeze(1).to(torch.float32)   # output size: [N, 1, 32, 15]
        x = self.conv1(x) # output size: [N, 64, 16, 15]
        x = self.conv2(x) # output size: [N, 128, 8, 15]
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.FC(x) # output size: [N, 2]
        x = self.Softmax(x)
        
        return x
    
    
    
class CNN20d(nn.Module):
    # Input: [N, (1), 64, 60]; Output: [N, 2]
    # Three Convolution Blocks
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN20d, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(1, 3), dilation=(2, 1))), # output size: [N, 64, 64, 20]
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=(2,1))) # output size: [N, 64, 32, 20]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 128, 32, 60]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=(2,1))) # output size: [N, 128, 16, 60]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 256, 16, 60]
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,2), stride=(2,2))) # output size: [N, 256, 8, 30]
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)
        
        # Additional pooling to match the required FC input size
        # 256 * 8 * 30 = 61440, we need 46080
        # 46080 / 256 = 180, so we need 8 * 180 = 1440
        # 需要调整架构来达到正确的尺寸
        self.extra_pool = nn.MaxPool2d((1, 1))  # 不进行额外池化，保持 [N, 256, 8, 30]

        self.DropOut = nn.Dropout(p=0.5)
        # 添加一个线性层来调整尺寸到46,080
        self.adjust_size = nn.Linear(61440, 46080)  # 从61,440调整到46,080
        self.FC = nn.Linear(46080, 2)  # 论文要求46,080个神经元
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x): # input: [N, 64, 60]
        x = x.unsqueeze(1).to(torch.float32)   # output size: [N, 1, 64, 60]
        print("After unsqueeze:", x.shape)
        x = self.conv1(x) # output size: [N, 64, 32, 60]
        print("After conv1:", x.shape)
        x = self.conv2(x) # output size: [N, 128, 16, 60]
        print("After conv2:", x.shape)
        x = self.conv3(x) # output size: [N, 256, 8, 30]
        print("After conv3:", x.shape)
        x = self.extra_pool(x) # output size: [N, 256, 8, 30]
        print("After extra_pool:", x.shape)
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.adjust_size(x)  # 调整尺寸到46,080
        x = self.FC(x) # output size: [N, 2]
        x = self.Softmax(x)
        
        return x


class CNN60d(nn.Module):
    # Input: [N, (1), 96, 180]; Output: [N, 2]
    # Four Convolution Blocks
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN60d, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(3, 1))), # output size: [N, 64, 96, 180]
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=2)) # output size: [N, 64, 48, 180]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 128, 48, 180]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=2)) # output size: [N, 128, 24, 180]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 256, 24, 180]
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=2)) # output size: [N, 256, 12, 180]
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)
        
        self.conv4 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(256, 512, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # output size: [N, 512, 12, 180]
            ('BN', nn.BatchNorm2d(512, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1), stride=2)) # output size: [N, 512, 6, 180]
        ]))
        self.conv4 = self.conv4.apply(self.init_weights)
        
        # Additional pooling to match the required FC input size of 184320
        # 512 * 6 * 180 = 552960, we need 184320
        # 184320 / 512 = 360, so we need 6 * 60 = 360
        # So we need one more pooling: 6 * 180 -> 6 * 60
        self.extra_pool = nn.MaxPool2d((1, 3))  # output size: [N, 512, 6, 60]

        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(184320, 2)  # 512 * 6 * 60 = 184320
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x): # input: [N, 96, 180]
        x = x.unsqueeze(1).to(torch.float32)   # output size: [N, 1, 96, 180]
        x = self.conv1(x) # output size: [N, 64, 48, 180]
        x = self.conv2(x) # output size: [N, 128, 24, 180]
        x = self.conv3(x) # output size: [N, 256, 12, 180]
        x = self.conv4(x) # output size: [N, 512, 6, 180]
        x = self.extra_pool(x) # output size: [N, 512, 6, 60]
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.FC(x) # output size: [N, 2]
        x = self.Softmax(x)
        
        return x
    