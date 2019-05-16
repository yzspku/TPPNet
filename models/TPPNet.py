import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

def SPP(x, pool_size):
    N, C, H, W = x.size()
    for i in range(len(pool_size)):
        maxpool = nn.AdaptiveMaxPool2d((H, pool_size[i]))
        if i==0: spp = maxpool(x).view(N, -1)
        else: spp = torch.cat((spp, maxpool(x).view(N, -1)),1)
    return spp


class CQT200Net(BasicModule):
    # No SPP Layer, input size is 1x84x200
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels= 32,kernel_size=(36,40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12,3), 
                                stride=(1,2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), 
                                stride=(1,2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            
            ('conv1', nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('conv2', nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.fc0 = nn.Linear(2048, 300) # ?
        self.fc1 = nn.Linear(300, 10000)
    def forward(self, x):
        # input [N, C, H, W]
        N = x.size()[0] # N, 1, 84, 400
        x = self.features(x) 
        x = self.conv(x) #  
        x = x.view(N,-1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature
class CQT300Net(BasicModule):
    # No SPP Layer, input size is 1x84x300
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels= 32,kernel_size=(36,40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12,3), 
                                stride=(1,2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), 
                                stride=(1,2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            
            ('conv1', nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('conv2', nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.fc0 = nn.Linear(3584, 300) # ?
        self.fc1 = nn.Linear(300, 10000)
    def forward(self, x):
        # input [N, C, H, W]
        N = x.size()[0] # N, 1, 84, 400
        x = self.features(x) 
        x = self.conv(x) #  
        x = x.view(N,-1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature
    
class CQT400Net(BasicModule):
    # No SPP Layer, input size is 1x84x400
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels= 32,kernel_size=(36,40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12,3), 
                                stride=(1,2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), 
                                stride=(1,2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            
            ('conv1', nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('conv2', nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.fc0 = nn.Linear(5120, 300) # ?
        self.fc1 = nn.Linear(300, 10000)
    def forward(self, x):
        # input [N, C, H, W]
        N = x.size()[0] # N, 1, 84, 400
        x = self.features(x) 
        x = self.conv(x) #  
        x = x.view(N,-1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature
    

class CQTTPPNet(BasicModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels= 32,kernel_size=(36,40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12,3), 
                                stride=(1,2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), 
                                stride=(1,2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            
            ('conv1', nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('conv2', nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(1,3), stride=(1,2),bias=False)),
            ('norm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

        self.fc0 = nn.Linear(10*512, 300)
        self.fc1 = nn.Linear(300, 10000)
    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        N = x.size()[0]
        x = self.features(x) # [N, 128, 1, W - 75 + 1]
        x = self.conv(x) #  [N, 256, 1, W - 75 +1 - 3 + 1]
        x = SPP(x, [4,3,2,1]) # [N, 256, 1, sum()=79]
        x = x.view(N,-1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature