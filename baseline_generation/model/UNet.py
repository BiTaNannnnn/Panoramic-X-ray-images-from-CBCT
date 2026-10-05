import torch
import random
# 针对图像处理的包
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# relu 激活函数
import torch.nn.functional as F
# 优化器的包
import torch.optim as optim
import matplotlib.pyplot as plt
from model.loss import *

########### ExtNet
class ExtNet(torch.nn.Module):
    def __init__(self):
        super(ExtNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.Tconv1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.Tconv2 = torch.nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        # BatchNorm2d（ channal ）
        self.BN = torch.nn.BatchNorm2d(64)
        self.BN_3 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        in_size = x.size(0)
        # Conv
        x = self.BN(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv3(x)))
        x = self.BN(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv3(x)))
        x = self.BN(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv3(x)))
        x = self.BN(F.relu(self.conv2(x)))
        print(x.shape)
        # TransCov = 反卷积
        x = self.BN(F.relu(self.conv3(x)))
        print(x.shape)
        x = self.BN(F.relu(self.Tconv1(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv3(x)))
        x = self.BN(F.relu(self.Tconv1(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv3(x)))
        x = self.BN(F.relu(self.Tconv1(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv3(x)))
        x = self.BN_3(F.relu(self.Tconv2(x)))
        print(x.shape)

        return x
model_extnet = ExtNet()
#
# ########### model - SegNet
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.name = 'Net_1'
        # self.conv1 = torch.nn.Conv2d(3,10,kernel_size=5)
        # self.conv2 = torch.nn.Conv2d(88,20,kernel_size=5)

        self.ExtNet = ExtNet()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.BN = torch.nn.BatchNorm2d(3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        in_size = x.size(0)
        x = self.ExtNet(x)
        print(x.shape)

        x = self.BN(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.BN(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.sigmoid(x)
        print(x.shape)
        # tensor_len = x.shape[1] * x.shape[2] * x.shape[3]
        # print("最后输出张量 = ",tensor_len)
        # x = x.view(in_size, -1)# flatten
        # x = self.fc(x)
        return x
# 实例化model
model = Net()

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 512, 512)
    # model_extnet(input_tensor)
    model(input_tensor)