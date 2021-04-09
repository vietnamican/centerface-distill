from models.mobilenetv2.mobilenetv2 import MobileNetV2Dense
import os
import os.path as osp
from time import time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetV2, MobileNetV2Dense
from datasets import WiderFace, WiderFaceVal


net = Model(MobileNetV2VGGBlockTemper1)
summary(net, (3, 416, 416))


