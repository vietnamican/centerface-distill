from torch import nn
import torch.utils.model_zoo as model_zoo

from ..base import Base, VGGBlock
from ..layers import ConvBNReLU, InvertedResidual, InvertedDenseResidual

__all__ = ['MobileNetV2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class MobileNetV2(Base):
    def __init__(self, width_mult=1.0, round_nearest=8,):
        super(MobileNetV2, self).__init__()
        self.feature_1 = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
        )
        self.feature_2 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
        )
        self.feature_4 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
        )
        self.feature_6 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 320, 1, 6),
        )

    def forward(self, x):
        y = []
        x = self.feature_1(x)
        y.append(x)
        x = self.feature_2(x)
        y.append(x)
        x = self.feature_4(x)
        y.append(x)
        y.append(self.feature_6(x))
        return y


class MobileNetV2Dense(Base):
    def __init__(self, width_mult=1.0, round_nearest=8,):
        super().__init__()
        self.feature_1 = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            InvertedDenseResidual(32, 16, 1, 1),
            InvertedDenseResidual(16, 24, 2, 2),
            InvertedDenseResidual(24, 24, 1, 2),
        )
        self.feature_2 = nn.Sequential(
            InvertedDenseResidual(24, 32, 2, 2),
            InvertedDenseResidual(32, 32, 1, 2),
            InvertedDenseResidual(32, 32, 1, 2),
        )
        self.feature_4 = nn.Sequential(
            InvertedDenseResidual(32, 64, 2, 2),
            InvertedDenseResidual(64, 64, 1, 2),
            InvertedDenseResidual(64, 64, 1, 2),
            InvertedDenseResidual(64, 64, 1, 2),
            InvertedDenseResidual(64, 96, 1, 2),
            InvertedDenseResidual(96, 96, 1, 2),
            InvertedDenseResidual(96, 96, 1, 2),
        )
        self.feature_6 = nn.Sequential(
            InvertedDenseResidual(96, 160, 2, 2),
            InvertedDenseResidual(160, 160, 1, 2),
            InvertedDenseResidual(160, 160, 1, 2),
            InvertedDenseResidual(160, 320, 1, 2),
        )

    def forward(self, x):
        y = []
        x = self.feature_1(x)
        y.append(x)
        x = self.feature_2(x)
        y.append(x)
        x = self.feature_4(x)
        y.append(x)
        y.append(self.feature_6(x))
        return y


class MobileNetV2VGGBlock(Base):
    def __init__(self, width_mult=1.0, round_nearest=8,):
        super().__init__()
        self.feature_1 = nn.Sequential(
            VGGBlock(3, 32, 2),
            VGGBlock(32, 16, 1),
            VGGBlock(16, 24, 2),
            VGGBlock(24, 24, 1),
        )
        self.feature_2 = nn.Sequential(
            VGGBlock(24, 32, 2),
            VGGBlock(32, 32, 1),
            VGGBlock(32, 32, 1),
        )
        self.feature_4 = nn.Sequential(
            VGGBlock(32, 64, 2),
            VGGBlock(64, 64, 1),
            VGGBlock(64, 64, 1),
            VGGBlock(64, 64, 1),
            VGGBlock(64, 96, 1),
            VGGBlock(96, 96, 1),
            VGGBlock(96, 96, 1),
        )
        self.feature_6 = nn.Sequential(
            VGGBlock(96, 160, 2),
            VGGBlock(160, 160, 1),
            VGGBlock(160, 160, 1),
            VGGBlock(160, 320, 1),
        )

    def forward(self, x):
        y = []
        x = self.feature_1(x)
        y.append(x)
        x = self.feature_2(x)
        y.append(x)
        x = self.feature_4(x)
        y.append(x)
        y.append(self.feature_6(x))
        return y

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()


class MobileNetV2VGGBlockTemper1(Base):
    def __init__(self, width_mult=1.0, round_nearest=8,):
        super().__init__()
        self.feature_1 = nn.Sequential(
            VGGBlock(3, 32, 2),
            VGGBlock(32, 16, 1),
            VGGBlock(16, 24, 2),
            VGGBlock(24, 24, 1),
        )
        self.feature_2 = nn.Sequential(
            VGGBlock(24, 32, 2),
            VGGBlock(32, 32, 1),
            VGGBlock(32, 32, 1),
        )
        self.feature_4 = nn.Sequential(
            VGGBlock(32, 64, 2),
            VGGBlock(64, 64, 1),
            VGGBlock(64, 64, 1),
            VGGBlock(64, 96, 1),
            VGGBlock(96, 96, 1),
        )
        self.feature_6 = nn.Sequential(
            VGGBlock(96, 160, 2),
            VGGBlock(160, 160, 1),
            VGGBlock(160, 320, 1),
        )

    def forward(self, x):
        y = []
        x = self.feature_1(x)
        y.append(x)
        x = self.feature_2(x)
        y.append(x)
        x = self.feature_4(x)
        y.append(x)
        y.append(self.feature_6(x))
        return y

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()
