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

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetV2, MobileNetV2Dense
from datasets import WiderFace, WiderFaceVal

traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                         cfg.downscale, cfg.insize, cfg.train_transforms, 'train')
trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                         pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

for im, hm, path in trainloader:
    print(path)
    print(hm.shape)
    hm = hm[1, 0]
    hm[hm > 0.5] = 1

    np.savetxt('x.txt', hm, fmt='%d')
    break
