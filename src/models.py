import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def get_activation(name: str = 'relu'):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'silu':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unknown activation: {name}')

class ClassificationModel(nn.Module):
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True, dropout: float = 0.3, activation: str = 'relu'):
        super().__init__()
        if backbone == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == 'resnet34':
            net = models.resnet34(pretrained=pretrained)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            net = models.efficientnet_b0(pretrained=pretrained)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            raise ValueError(f'Unknown backbone: {backbone}')
        self.backbone = net
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
        self.activation = get_activation(activation)
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.squeeze(-1)

# U-Net for segmentation
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, activation: str = 'relu'):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64, activation)
        self.enc2 = self.conv_block(64, 128, activation)
        self.enc3 = self.conv_block(128, 256, activation)
        self.enc4 = self.conv_block(256, 512, activation)
        self.pool = nn.MaxPool2d(2)
        self.center = self.conv_block(512, 1024, activation)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512, activation)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256, activation)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128, activation)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64, activation)
        self.final = nn.Conv2d(64, out_channels, 1)
    def conv_block(self, in_c, out_c, activation):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            get_activation(activation),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            get_activation(activation),
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        out = self.final(d1)
        return out
