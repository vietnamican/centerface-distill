from models.mobilenetv2 import MobileNetV2, MobileNetV2VGGBlock
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger

from config import Config as cfg
from datasets import WiderFace
from models.mobile_net_temper_wrapper import MobileNetTemperWrapper

# Data Setup
traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                         cfg.downscale, cfg.insize, cfg.train_transforms, 'train')
trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                         pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
valdataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                       cfg.downscale, cfg.insize, cfg.train_transforms, 'val')
valloader = DataLoader(valdataset, batch_size=cfg.batch_size,
                       pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

orig_module_names = [
    'orig.feature_1.0',
    'orig.feature_1.1',
    'orig.feature_1.2',
    'orig.feature_1.3',
    'orig.feature_2.0',
    'orig.feature_2.1',
    'orig.feature_2.2',
    'orig.feature_3.0',
    'orig.feature_3.1',
    'orig.feature_3.2',
    'orig.feature_3.3',
    'orig.feature_3.4',
    'orig.feature_3.5',
    'orig.feature_3.6',
    'orig.feature_4.0',
    'orig.feature_4.1',
    'orig.feature_4.2',
    'orig.feature_4.3',
]

tempered_module_names = [
    'tempered.feature_1.0',
    'tempered.feature_1.1',
    'tempered.feature_1.2',
    'tempered.feature_1.3',
    'tempered.feature_2.0',
    'tempered.feature_2.1',
    'tempered.feature_2.2',
    'tempered.feature_3.0',
    'tempered.feature_3.1',
    'tempered.feature_3.2',
    'tempered.feature_3.3',
    'tempered.feature_3.4',
    'tempered.feature_3.5',
    'tempered.feature_3.6',
    'tempered.feature_4.0',
    'tempered.feature_4.1',
    'tempered.feature_4.2',
    'tempered.feature_4.3',
]

is_trains = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
]

if __name__ == '__main__':
    mode = 'temper'
    orig = MobileNetV2()
    tempered = MobileNetV2VGGBlock()
    if mode == 'temper':
        log_name = 'centerface_logs/{}'.format(mode)
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
        model = MobileNetTemperWrapper(
            orig, tempered, mode, orig_module_names, tempered_module_names, is_trains)

        # model.orig.migrate()
        trainer = pl.Trainer(
            max_epochs=70,
            logger=logger,
            callbacks=callbacks,
            gpus=1
        )
        trainer.fit(model, trainloader, valloader)