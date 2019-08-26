import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import floor
from operator import mul


def Inception3(input_size=(32,32), num_classes=10):
    model = torchvision.models.inception.Inception3(num_classes=num_classes)
    model.name = "Inception3"
    return model

def ResNet10(input_size=(32, 32), num_classes=10):
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    model.name = "ResNet10"
    return model

def ResNet12(input_size=(32, 32), num_classes=10):
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [1, 1, 2, 1], num_classes=num_classes)
    model.name = "ResNet12"
    return model

def ResNet14(input_size=(32, 32), num_classes=10):
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [1, 2, 2, 1], num_classes=num_classes)
    model.name = "ResNet14"
    return model

def ResNet16(input_size=(32, 32), num_classes=10):
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [1, 2, 2, 2], num_classes=num_classes)
    model.name = "ResNet16"
    return model

def ResNet18(input_size=(32,32), num_classes=10):
    model = torchvision.models.resnet.resnet18(num_classes=num_classes)
    model.name = "ResNet18"
    return model

def ResNet34(input_size=(32,32), num_classes=10):
    model = torchvision.models.resnet.resnet34(num_classes=num_classes)
    model.name = "ResNet34"
    return model

def ResNet50(input_size=(32,32), num_classes=10):
    model = torchvision.models.resnet.resnet50(num_classes=num_classes)
    model.name = "ResNet34"
    return model

def ResNet101(input_size=(32,32), num_classes=10):
    model = torchvision.models.resnet.resnet101(num_classes=num_classes)
    model.name = "ResNet101"
    return model

def ResNet152(input_size=(32,32), num_classes=10):
    model = torchvision.models.resnet.resnet152(num_classes=num_classes)
    model.name = "ResNet152"
    return model

class LeNet(nn.Module):
    name = "LeNet"

    @staticmethod
    def _input_fc_size(input_size: int):
        conv_size1 = input_size - 2
        pool_size1 = floor((conv_size1 - 2) / 2) + 1
        conv_size2 = pool_size1 - 2
        pool_size2 = floor((conv_size2 - 2) / 2) + 1
        return pool_size2

    def __init__(self, input_size=(512,512), num_classes=2):
        super(LeNet, self).__init__()
        self.input_size = input_size
        self.input_fc_dims = tuple(map(self._input_fc_size, input_size))
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * mul(*self.input_fc_dims), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * mul(*self.input_fc_dims))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    'BS': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'BXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'BXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'BXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'DS': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'DXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'],
    'DXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M'],
    'DXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'ES': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    'EXS': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 128, 128, 128, 128, 'M'],
    'EXXS': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64, 64, 64, 64, 'M'],
    'EXXXS': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 32, 32, 32, 32, 'M'],
}


def make_layers(cfg, batch_norm=True, k_size=3):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=k_size, padding=k_size-2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True,
                 final_filter: int = 512, pretrained=False,
                 input_size=(32,32)):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(final_filter),
            nn.Dropout(0.25),
            nn.Linear(final_filter, final_filter//2),
            nn.ReLU(True),
            nn.BatchNorm1d(final_filter//2),
            nn.Dropout(0.25),
            nn.Linear(final_filter//2, num_classes)
        )
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
    model.name = "VGG16"
    return model

def vgg16_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DS']), final_filter=256, **kwargs)
    model.name = "VGG16_S"
    return model


def vgg16_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXS']), final_filter=128, **kwargs)
    model.name = "VGG16_XS"
    return model


def vgg16_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXXS']), final_filter=64, **kwargs)
    model.name = "VGG16_XXS"
    return model


def vgg16_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['DXXXS']), final_filter=32, **kwargs)
    model.name = "VGG16_XXXS"
    return model


def vgg16_5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=5), **kwargs)
    model.name = "VGG16_5"
    return model


def vgg16_7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], k_size=7), **kwargs)
    model.name = "VGG16_7"
    return model


def vgg5(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['V']), final_filter=64, **kwargs)
    model.name = "VGG5"
    return model


def vgg5_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['VS']), final_filter=32, **kwargs)
    model.name = "VGG5_S"
    return model


def vgg6(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['W']), final_filter=128, **kwargs)
    model.name = "VGG6"
    return model


def vgg6_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['WS']), final_filter=64, **kwargs)
    model.name = "VGG6_S"
    return model


def vgg7(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['X']), final_filter=256, **kwargs)
    model.name = "VGG7"
    return model


def vgg7_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['XS']), final_filter=128, **kwargs)
    model.name = "VGG7_S"
    return model


def vgg8(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Y']), **kwargs)
    model.name = "VGG8"
    return model


def vgg8_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['YS']), final_filter=256, **kwargs)
    model.name = "VGG8_S"
    return model


def vgg8_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['YXS']), final_filter=128, **kwargs)
    model.name = "VGG8_XS"
    return model


def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z']), **kwargs)
    model.name = "VGG9"
    return model


def vgg9_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZS']), final_filter=256, **kwargs)
    model.name = "VGG9_S"
    return model


def vgg9_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZXS']), final_filter=128, **kwargs)
    model.name = "VGG9_XS"
    return model


def vgg9_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ZXXS']), final_filter=64, **kwargs)
    model.name = "VGG9_XXS"
    return model


def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['Z']), **kwargs)
    model.name = "VGG9"
    return model

def vgg11(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    model.name = "VGG11"
    return model

def vgg11_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AS']), final_filter=256, **kwargs)
    model.name = "VGG11_S"
    return model

def vgg11_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXS']), final_filter=128, **kwargs)
    model.name = "VGG11_XS"
    return model

def vgg11_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXXS']), final_filter=64, **kwargs)
    model.name = "VGG11_XXS"
    return model

def vgg11_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['AXXXS']), final_filter=32, **kwargs)
    model.name = "VGG11_XXXS"
    return model


def vgg13(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    model.name = "VGG13"
    return model

def vgg13_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BS']), final_filter=256, **kwargs)
    model.name = "VGG13_S"
    return model

def vgg13_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXS']), final_filter=128, **kwargs)
    model.name = "VGG13_XS"
    return model

def vgg13_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXXS']), final_filter=64, **kwargs)
    model.name = "VGG13_XXS"
    return model

def vgg13_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['BXXXS']), final_filter=32, **kwargs)
    model.name = "VGG13_XXXS"
    return model

def vgg19(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    model.name = "VGG19"
    return model

def vgg19_S(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['ES']), final_filter=256, **kwargs)
    model.name = "VGG19_S"
    return model

def vgg19_XS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXS']), final_filter=128, **kwargs)
    model.name = "VGG19_XS"
    return model

def vgg19_XXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXXS']), final_filter=64, **kwargs)
    model.name = "VGG19_XXS"
    return model

def vgg19_XXXS(*args, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['EXXXS']), final_filter=32, **kwargs)
    model.name = "VGG19_XXXS"
    return model
