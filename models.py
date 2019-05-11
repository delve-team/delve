import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from fastai.layers import AdaptiveConcatPool2d
from delve import CheckLayerSat
from tqdm import tqdm, trange
from fastai.vision import learner


class SimpleFCNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 l1: int = 1024,
                 l2: int = 512,
                 l3: int = 256,
                 n_classes: int = 10):
        super(SimpleFCNet, self).__init__()

        print('Setting up FCN with: l1', l1, 'l2', l2, 'l3', l3)

        # feature extractor
        self.fc0 = nn.Linear(in_channels, l1)
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        # readout + head
        self.fc3 = nn.Linear(l3, 128)
        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 l1: int = 8,
                 l2: int = 16,
                 l3: int = 32,
                 n_classes: int = 10):
        super(SimpleCNN, self).__init__()

        print('Setting up CNN with: l1', l1, 'l2', l2, 'l3', l3)

        # feature exxtractor
        self.conv00 = nn.Conv2d(in_channels=in_channels,
                                out_channels=l1,
                                kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv10 = nn.Conv2d(in_channels=l1, out_channels=l2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv20 = nn.Conv2d(in_channels=l2, out_channels=l3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2, 2)
        # readout + head
        # self.pool = AdaptiveConcatPool2d(1)
        self.fc0 = nn.Linear(l3 * 400, l3)
        #   self.fc1 = nn.Linear(l3, l3//2)
        self.out = nn.Linear(l3, n_classes)

    def forward(self, x):
        x = F.relu(self.conv00(x))
        #x = self.pool1(x)
        x = F.relu(self.conv10(x))
        #x = self.pool2(x)
        x = F.relu(self.conv20(x))
        #x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        # x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class SimpleCNNKernel(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 l1: int = 5,
                 l2: int = 5,
                 l3: int = 5,
                 n_classes: int = 10):
        super(SimpleCNNKernel, self).__init__()

        print('Setting up CNN with: kernel1', l1, 'kernel2', l2, 'kernel3', l3)

        out_res = 32 - (2 * (l1 // 2)) - (2 * (l2 // 2)) - (2 * (l3 // 2))
        out_res = out_res**2 * 32

        # feature exxtractor
        self.conv00 = nn.Conv2d(in_channels=in_channels,
                                out_channels=8,
                                kernel_size=l1)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=l2)
        self.conv20 = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=l3)
        # readout + head
        # self.pool = AdaptiveConcatPool2d(1)
        self.fc0 = nn.Linear(out_res, l3)
        #   self.fc1 = nn.Linear(l3, l3//2)
        self.out = nn.Linear(l3, n_classes)

    def forward(self, x):
        x = F.relu(self.conv00(x))
        #x = self.pool1(x)
        x = F.relu(self.conv10(x))
        #x = self.pool2(x)
        x = F.relu(self.conv20(x))
        #x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        # x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


cfg = {
    'V': [64, 'M'],
    'VS': [32, 'M'],
    'W': [64, 'M', 128, 'M'],
    'WS': [32, 'M', 64, 'M'],
    'X': [64, 'M', 128, 'M', 256, 'M'],
    'XS': [32, 'M', 64, 'M', 128, 'M'],
    'Y': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'YS': [32, 'M', 64, 'M', 128, 'M', 256, 'M'],
    'YXS': [16, 'M', 32, 'M', 64, 'M', 128, 'M'],
    'Z': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'ZXS': [16, 'M', 32, 'M', 64, 'M', 128, 'M', 128, 'M'],
    'ZXXS': [8, 'M', 16, 'M', 32, 'M', 64, 'M', 64, 'M'],
    'ZXXXS': [4, 'M', 8, 'M', 16, 'M', 32, 'M', 32, 'M'],
    'ZS': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 256, 'M'],
    'AS': [32, 'M', 64, 'M', 126, 126, 'M', 256, 256, 'M', 256, 256, 'M'],
    'AXS': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'AXXS': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'AXXXS': [4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'AL':
    [128, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 1024, 'M'],
    'BS':
    [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'BXS':
    [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'BXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'BXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'DS': [
        32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256,
        256, 256, 'M'
    ],
    'DXXXS':
    [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'],
    'DXXS': [
        8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64,
        'M'
    ],
    'DXS': [
        16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128,
        128, 128, 'M'
    ],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'DL': [
        128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024,
        'M', 1024, 1024, 1024, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
    'ES': [
        32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256,
        'M', 256, 256, 256, 256, 'M'
    ],
    'EXS': [
        16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M',
        128, 128, 128, 128, 'M'
    ],
    'EXXS': [
        8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64,
        64, 64, 64, 'M'
    ],
    'EXXXS': [
        4, 4, 'M', 8, 8, 'M', 16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 32, 32,
        32, 32, 'M'
    ],
}


def make_layers(cfg, batch_norm=True, k_size=3):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,
                               v,
                               kernel_size=k_size,
                               padding=k_size - 2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self,
                 features,
                 num_classes=10,
                 init_weights=True,
                 final_filter: int = 512,
                 pretrained=False):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(final_filter), nn.Dropout(0.25),
            nn.Linear(final_filter, final_filter // 2), nn.ReLU(True),
            nn.BatchNorm1d(final_filter // 2), nn.Dropout(0.25),
            nn.Linear(final_filter // 2, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_L(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DL']), **kwargs)
    return model


def vgg16_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DS']), final_filter=256, **kwargs)
    return model


def vgg16_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXS']), final_filter=128, **kwargs)
    return model


def vgg16_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXXS']), final_filter=64, **kwargs)
    return model


def vgg16_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXXXS']), final_filter=32, **kwargs)
    return model


def vgg16_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=5), **kwargs)
    return model


def vgg16_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=7), **kwargs)
    return model


def vgg5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['V']), final_filter=64, **kwargs)
    return model


def vgg5_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['VS']), final_filter=32, **kwargs)
    return model


def vgg6(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['W']), final_filter=128, **kwargs)
    return model


def vgg6_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['WS']), final_filter=64, **kwargs)
    return model


def vgg7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['X']), final_filter=256, **kwargs)
    return model


def vgg7_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['XS']), final_filter=128, **kwargs)
    return model


def vgg8(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Y']), **kwargs)
    return model


def vgg8_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['YS']), final_filter=256, **kwargs)
    return model


def vgg8_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['YXS']), final_filter=128, **kwargs)
    return model


def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z']), **kwargs)
    return model


def vgg9_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZS']), final_filter=256, **kwargs)
    return model


def vgg9_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZXS']), final_filter=128, **kwargs)
    return model


def vgg9_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZXXS']), final_filter=64, **kwargs)
    return model


def vgg9_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z'], k_size=5), **kwargs)
    return model


def vgg9_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z'], k_size=7), **kwargs)
    return model


def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z']), **kwargs)
    return model


def vgg11(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_L(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AL']), **kwargs)
    return model


def vgg11_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AS']), final_filter=256, **kwargs)
    return model


def vgg11_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXS']), final_filter=128, **kwargs)
    return model


def vgg11_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXXS']), final_filter=64, **kwargs)
    return model


def vgg11_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXXXS']), final_filter=32, **kwargs)
    return model


def vgg11_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], k_size=5), **kwargs)
    return model


def vgg11_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], k_size=7), **kwargs)
    return model


def vgg11nbn(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=False), **kwargs)
    return model


def vgg11nbn_L(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AL'], batch_norm=False), **kwargs)
    return model


def vgg11nbn_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AS'], batch_norm=False),
                final_filter=256,
                **kwargs)
    return model


def vgg11nbn_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], k_size=5, batch_norm=False), **kwargs)
    return model


def vgg11nbn_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], k_size=7, batch_norm=False), **kwargs)
    return model


def vgg13(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BS']), final_filter=256, **kwargs)
    return model


def vgg13_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXS']), final_filter=128, **kwargs)
    return model


def vgg13_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXXS']), final_filter=64, **kwargs)
    return model


def vgg13_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXXXS']), final_filter=32, **kwargs)
    return model


def vgg13_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], k_size=5), **kwargs)
    return model


def vgg13_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], k_size=7), **kwargs)
    return model


def vgg19(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ES']), final_filter=256, **kwargs)
    return model


def vgg19_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXS']), final_filter=128, **kwargs)
    return model


def vgg19_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXXS']), final_filter=64, **kwargs)
    return model


def vgg19_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXXXS']), final_filter=32, **kwargs)
    return model


def vgg19_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], k_size=5), **kwargs)
    return model


def vgg19_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], k_size=5), **kwargs)
    return model


def vgg19_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], k_size=7), **kwargs)
    return model
