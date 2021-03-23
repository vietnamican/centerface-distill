import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1
from datasets import WiderFace, WiderFaceVal

device = 'cpu'
checkpoint_path = 'temp.ckpt'

net = Model(MobileNetV2VGGBlockTemper1)
if device == 'cpu' or device == 'tpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint
net.migrate(state_dict, force=True, verbose=2)