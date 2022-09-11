import torch.nn as nn
import torch
from torchvision import models
from torch import autograd

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


class Unet_resnet(nn.Module):
    def __init__(self, num_classes=1):
        super(Unet_resnet, self).__init__()

        # filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.up6 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv6 = DoubleConv(2048, 1024)
        self.up7 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv7 = DoubleConv(1024, 512)
        self.up8 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv8 = DoubleConv(512, 256)
        self.up9 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv9 = DoubleConv(192, 128)
        self.up10 = nn.ConvTranspose2d(128,64, 2, stride=2)
        self.conv10 = DoubleConv(64,64)
        self.finalconv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = x
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        up_6 = self.up6(e4)
        # print(up_6.shape)
        merge6 = torch.cat([up_6, e3], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        # print(up_7.shape)
        merge7 = torch.cat([up_7, e2], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        # print(up_8.shape)
        merge8 = torch.cat([up_8, e1], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        # print(up_9.shape)
        merge9 = torch.cat([up_9, e0], dim=1)
        # print(merge9.shape)
        c9 = self.conv9(merge9)
        # print(c9.shape)
        up_10 = self.up10(c9)
        c10 = self.conv10(up_10)
        c11 = self.finalconv(c10)
        out = nn.Sigmoid()(c11)
        # print(out.shape)
        return out