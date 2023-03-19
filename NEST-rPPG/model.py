# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
import numpy as np

np.set_printoptions(threshold=np.inf)
sys.path.append('..')


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
                 )
        self.downsample = downsample
        self.Res = Res

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        return F.relu(out)



class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        model_resnet = models.resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [2, 1], downsample=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 1, [1, 1], downsample=1),
        )

    def get_av(self, x):
        av = torch.mean(torch.mean(x, dim=-1), dim=-1)
        min, _ = torch.min(av, dim=1, keepdim=True)
        max, _ = torch.max(av, dim=1, keepdim=True)
        av = torch.mul((av-min),((max-min).pow(-1)))
        return av
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        av1 = self.get_av(x)
        x = self.layer2(x)
        av2 = self.get_av(x)
        x = self.layer3(x)
        av3 = self.get_av(x)
        em = self.layer4(x)
        av4 = self.get_av(em)

        av = torch.cat([av1, av2, av3, av4], dim=1)

        HR = self.fc(self.avgpool(em).view(x.size(0), -1))
        # For Sig
        x = self.up1(em)
        x = self.up2(x)
        x = self.up3(x)
        Sig = self.up4(x).squeeze(dim=1)


        return Sig, HR, av