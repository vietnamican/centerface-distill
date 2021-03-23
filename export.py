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
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetTemperWrapper, configs
from datasets import WiderFace

device = 'cpu'

config = configs[1]
net = MobileNetTemperWrapper(
    config['orig'](),
    config['tempered'](),
    'tuning',
    config['orig_module_names'],
    config['tempered_module_names'],
    config['is_trains'],
)
# checkpoint_path = 'centerface_logs/temper/version_2/checkpoints/checkpoint-epoch=53-val_loss=2.8531.ckpt'
# if device == 'cpu' or device == 'tpu':
#     checkpoint = torch.load(
#         checkpoint_path, map_location=lambda storage, loc: storage)
# else:
#     checkpoint = torch.load(checkpoint_path)
# state_dict = checkpoint['state_dict']
# net.migrate(state_dict, force=True, verbose=2)
net.export('temp.ckpt')
