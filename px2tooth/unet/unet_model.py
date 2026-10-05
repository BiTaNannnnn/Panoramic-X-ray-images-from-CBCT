""" Full assembly of the parts to form the complete network

Unet = 全景片分割网络
"""
""" https://www.bilibili.com/video/BV1rq4y1w7xM?p=1&vd_source=62bb03bf2697929fe9adfa731eae67ae"""
from .unet_parts import *

#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.down1 = DownsamplingBlock(n_channels, 64)
#         self.down2 = DownsamplingBlock(64, 128)
#         self.down3 = DownsamplingBlock(128, 256)
#         self.down4 = DownsamplingBlock(256, 512)
#
#         self.up1 = UpsamplingBlock(512, 256)
#         self.up2 = UpsamplingBlock(256, 128)
#         self.up3 = UpsamplingBlock(128, 64)
#         self.up4 = UpsamplingBlock(64, 64)
#
#         self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
#
#     def forward(self, x):
#         x = self.down1(x)
#         x = self.down2(x)
#         x = self.down3(x)
#         x = self.down4(x)
#
#         x = self.up1(x)
#         x = self.up2(x)
#         x = self.up3(x)
#         x = self.up4(x)
#
#         logits = self.outc(x)
#         return logits
#
#
# class DownsamplingBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownsamplingBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         return x
#
# class UpsamplingBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UpsamplingBlock, self).__init__()
#         self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv_transpose(x)
#         x = self.relu(x)
#         # x = self.conv1(x)
#         # x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         return x




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        # bilinear = 是否采用双线性差值
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class UNet_3d(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        # bilinear = 是否采用双线性差值
        super(UNet_3d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Downsample
        self.conv1 = (DoubleConv(n_channels, 64))
        self.conv2 = (DoubleConv(64, 128))
        self.conv3 = (DoubleConv(128, 256))
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(256 * 8 * 8, 256 * 8 * 8)
        self.upconv1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv4 = nn.Conv3d(128, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv5 = nn.Conv3d(64, 64, 3, padding=1)
        self.upconv3 = nn.ConvTranspose3d(64, n_classes, 2, stride=2)
        self.conv6 = nn.Conv3d(n_classes, n_classes, 1)

    def forward(self, x):
        x = self.conv1(x)  # [1,64,512,512]
        x = self.pool(x)  # [1,64,256,256]
        x = self.conv2(x)  # [1,128,256,256]
        x = self.pool(x)  # [1,128,128,128]
        x = self.conv3(x)  # [1,256,128,128]
        x = self.pool(x)  # [1,256,64,64]
        x = x.reshape(-1, 256 * 8 * 8)  # [64，16384]
        x = torch.relu(self.fc(x))
        x = x.reshape(1, 256, 64, 64)  # 三维
        x = self.upconv1(x.unsqueeze(-1)).squeeze(-1)
        x = torch.relu(self.conv4(x))  # [1, 128, 128, 128, 2]
        x = self.upconv2(x)  # [1, 64, 256, 256, 4]
        x = torch.relu(self.conv5(x))
        x = self.upconv3(x)  # [1, 41, 512, 512, 8]
        x = torch.sigmoid(self.conv6(x))  # [1, 41, 512, 512, 8]
        # change tensor
        new_size = (1, 60606, 3)  # 体素化 512512512
        # [1, 41, 1, 60606, 3]
        x = F.interpolate(x, size=new_size, mode='trilinear', align_corners=False)  # mode = 插值方式
        print(x.shape)
        x = x.squeeze(2)  # [1, 41, 60606, 3]
        return x

    def use_checkpointing(self):
        self.conv1 = torch.utils.checkpoint(self.conv1)
        self.conv2 = torch.utils.checkpoint(self.conv2)
        self.conv3 = torch.utils.checkpoint(self.conv3)
        self.pool = torch.utils.checkpoint(self.pool)
        self.fc = torch.utils.checkpoint(self.fc)
        self.upconv1 = torch.utils.checkpoint(self.upconv1)
        self.upconv2 = torch.utils.checkpoint(self.upconv2)
        self.upconv3 = torch.utils.checkpoint(self.upconv3)
        self.conv4 = torch.utils.checkpoint(self.conv4)
        self.conv5 = torch.utils.checkpoint(self.conv5)
        self.conv6 = torch.utils.checkpoint(self.conv6)