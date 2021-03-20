import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config as cfg
from models.loss import RegLoss
from models.model import Model
from datasets import WiderFace


# Data Setup
traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                         cfg.downscale, cfg.insize, cfg.train_transforms, 'train')
trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                         pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
valdataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                       cfg.downscale, cfg.insize, cfg.train_transforms, 'val')
valloader = DataLoader(valdataset, batch_size=cfg.batch_size,
                       pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
device = cfg.device

# Network Setup
net = Model()
checkpoint_path = 'checkpoints/final.pth'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
net.migrate(checkpoint, force=True, verbose=2)

log_name = 'centerface/training'
logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    name=log_name,
    # log_graph=True,
    # version=0
)

loss_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='',
    filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    save_top_k=-1,
    mode='min',
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
callbacks = [loss_callback, lr_monitor]

trainer = pl.Trainer(
    max_epochs=140,
    logger=logger,
    callbacks=callbacks,
    gpus=1
)

trainer.fit(net, trainloader, valloader)
