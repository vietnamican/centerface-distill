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
from models.mobilenetv2 import MobileNetTemperWrapper, configs

pl.seed_everything(42)

# Data Setup
traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                         cfg.downscale, cfg.insize, cfg.train_transforms, 'train')
trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                         pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
valdataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                       cfg.downscale, cfg.insize, cfg.train_transforms, 'val')
valloader = DataLoader(valdataset, batch_size=cfg.batch_size,
                       pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

device = 'gpu'

if __name__ == '__main__':
    mode = 'temper'
    other_information = ''
    if len(other_information) > 0:
        mode = '{}_{}'.format(mode, other_information)
    config = configs[0]
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
            config['orig'](), config['tempered'](), mode, config["orig_module_names"], config["tempered_module_names"], config["is_trains"])
        
        checkpoint_path = 'checkpoints/final.pth'
        if device == 'cpu' or device == 'tpu':
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(checkpoint_path)
        # state_dict = checkpoint['state_dict']
        state_dict = checkpoint
        model.orig.migrate(state_dict, force=True, verbose=2)
        if device == 'tpu':
            trainer = pl.Trainer(
                max_epochs=10,
                logger=logger,
                callbacks=callbacks,
                tpu_cores=8
            )
        elif device == 'gpu':
            trainer = pl.Trainer(
                max_epochs=10,
                logger=logger,
                callbacks=callbacks,
                gpus=1
            )
        else:
            trainer = pl.Trainer(
                max_epochs=10,
                logger=logger,
                callbacks=callbacks
            )
        trainer.fit(model, trainloader, valloader)
