from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math


from .base import Base
from .layers import FPN
from .mobilenetv2 import MobileNetV2


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def load_model(model, state_dict):
    new_model = model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)


class CenterFace(Base):
    def __init__(self, head_conv=24, pretrained=True):
        super(CenterFace, self).__init__()
        self.heads = {'hm': 1, 'wh': 2, 'lm': 10, 'off': 2}
        # self.base = MobileNetV2()
        channels = [24, 32, 96, 320]
        self.fpn = FPN(channels, out_dim=head_conv)
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Conv2d(head_conv, classes,
                           kernel_size=1, stride=1,
                           padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                nn.init.normal_(fc.weight, std=0.001)
                nn.init.constant_(fc.bias, 0)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.fpn(x)
        # ret = {}
        return [self.hm(x), self.off(x), self.wh(x), self.lm(x)]


# def mobilenetv2_10(pretrained=True, **kwargs):
#     model = MobileNetV2(width_mult=1.0)
#     if pretrained:
#         state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
#                                               progress=True)
#         load_model(model,state_dict)
#     return model

# def mobilenetv2_5(pretrained=False, **kwargs):
#     model = MobileNetV2(width_mult=0.5)
#     if pretrained:
#         print('This version does not have pretrain weights.')
#     return model

# # num_layers  : [10 , 5]
# def get_mobile_net(num_layers, heads, head_conv=24):
#   model = MobileNetSeg('mobilenetv2_{}'.format(num_layers), heads,
#                  pretrained=True,
#                  head_conv=head_conv)
#   return model
