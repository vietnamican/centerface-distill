import torch
from torch import nn
import pytorch_lightning as pl

from .mnet import MobileNetSeg, MobileNetV2
from .base import Base


class Model(MobileNetSeg):
    def __init__(self, base):
        super().__init__(base, {'hm':1, 'wh':2, 'lm':10, 'off':2})
