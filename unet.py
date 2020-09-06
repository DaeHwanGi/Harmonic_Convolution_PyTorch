import torch
import torch.nn as nn
from harmonic_conv import HarmonicConv

def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
def ConvBlock(conv_layer, in_channels, out_channels):
    return nn.Sequential(
        conv_layer(in_channels, in_channels, 7, padding=3),
        nn.InstanceNorm2d(in_channels),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, 7, padding=3),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, conv_type='regular'):
        super().__init__()

        if conv_type is 'regular' or conv_type is 'dilated':
            conv_layer = nn.Conv2d
        elif conv_type is 'harmonic':
            conv_layer = HarmonicConv

        self.encode_layer1 = ConvBlock(nn.Conv2d, 2, 35)
        self.encode_layer2 = ConvBlock(conv_layer, 35, 70)
        self.encode_layer3 = ConvBlock(conv_layer, 70, 70)

        self.decode_layer1 = ConvBlock(conv_layer, 140, 35)
        self.decode_layer2 = ConvBlock(conv_layer, 70, 35)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_last = nn.Conv2d(35, 2, 1)
        self.apply(init_weight)

    def forward(self, x):
        x = x.unsqueeze(0)
        x1 = self.encode_layer1(x)
        x2 = self.encode_layer2(self.maxpool(x1))
        x3 = self.encode_layer3(self.maxpool(x2))
        x3 = self.upsample(x3)
        
        x = self.decode_layer1(torch.cat([x3, x2], dim=1))
        x = self.upsample(x)
        x = self.decode_layer2(torch.cat([x, x1], dim=1))
        x = self.conv_last(x)
        x = x.squeeze(0)
        return x
