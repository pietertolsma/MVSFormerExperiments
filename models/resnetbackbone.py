import torch
import torch.nn as nn
from torchvision.models import resnet50

import os
import shutil
import time
import numpy as np

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.args = args

        self.vit_chs = args["vit_ch"]

        self.prelayer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        ) # 256

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.inner1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.inner2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.inner3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # self.featuremap_count = 0
        # self.featuremap_limit = 20
        # if os.path.exists("resnet-features"):
        #     shutil.rmtree("resnet-features")

    def forward(self, x):
        x = self.prelayer(x)
        if x.isnan().any():
            print("x isnan")

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        down1 = self.smooth1(
            self.upsample1(out4) + 
            self.inner1(out3)
        )

        down2 = self.smooth2(
            self.upsample2(down1) + 
            self.inner2(out2)
        )

        down3 = self.smooth3(
            self.upsample3(down2) +
            self.inner3(out1)
        )

        if down3.isnan().any():
            print("down3 isnan")


        # if self.featuremap_limit > self.featuremap_count:
        #     t0 = time.time_ns()
        #     os.makedirs(f"resnet-features/{t0}", exist_ok=True)
        #     np.save(f"resnet-features/{t0}/x1.npy", down1.detach().cpu().numpy())
        #     np.save(f"resnet-features/{t0}/x2.npy", down2.detach().cpu().numpy())
        #     np.save(f"resnet-features/{t0}/x3.npy", down3.detach().cpu().numpy())

        #     self.featuremap_count += 1

        #out4 torch.Size([6, 2048, 12, 12]) down1 torch.Size([6, 256, 24, 24]) down2 torch.Size([6, 128, 48, 48]) down3 torch.Size([6, 64, 96, 96])

        return out4, down1, down2, down3