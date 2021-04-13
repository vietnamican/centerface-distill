from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math

from .base import Base, ConvBatchNormRelu, ConvTransposeBatchNormRelu
from .layers import IDAUp


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MobileNetUp(nn.Module):
    def __init__(self, channels, out_dim=24):
        super(MobileNetUp, self).__init__()
        channels = channels[::-1]
        self.conv = ConvBatchNormRelu(channels[0], out_dim,
                                      kernel_size=1, stride=1, bias=False)
        self.conv_last = ConvBatchNormRelu(out_dim, out_dim,
                                           kernel_size=3, stride=1, padding=1, bias=False)

        for i, channel in enumerate(channels[1:]):
            setattr(self, 'up_%d' % (i), IDAUp(out_dim, channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        x = self.conv(layers[-1])
        for i in range(0, len(layers)-1):
            up = getattr(self, 'up_{}'.format(i))
            x = up([x, layers[len(layers)-2-i]])
        x = self.conv_last(x)
        return x


class MobileNetSeg(Base):
    def __init__(self, base, heads, head_conv=24, pretrained=True):
        super(MobileNetSeg, self).__init__()
        self.heads = heads
        self.base = base()
        channels = [24, 32, 96, 320]
        self.dla_up = MobileNetUp(channels, out_dim=head_conv)
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Conv2d(head_conv, classes,
                           kernel_size=1, stride=1,
                           padding=0, bias=True)
            # if 'hm' in head:
            #     fc.bias.data.fill_(-2.19)
            #     fc = nn.Sequential(fc, nn.Sigmoid())
            # else:
            # nn.init.normal_(fc.weight, std=0.001)
            # nn.init.constant_(fc.bias, 0)
            if 'hm' in head:
                fc = nn.Sequential(fc, nn.Sigmoid())
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        ret = []
        for head in self.heads:
            ret.append(self.__getattr__(head)(x))
        return ret
