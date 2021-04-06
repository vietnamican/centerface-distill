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
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2, MobileNetV2Dense
from datasets import WiderFace

# Data Setup
traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                    cfg.downscale, cfg.insize, cfg.train_transforms, 'easy')
trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

valdataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                    cfg.downscale, cfg.insize, cfg.test_transforms, 'easy')
valloader = DataLoader(valdataset, batch_size=cfg.batch_size,
                        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

testdataset = WiderFace(cfg.valdataroot, cfg.valannfile, cfg.sigma,
                         cfg.downscale, cfg.insize, cfg.test_transforms, 'easy')
testloader = DataLoader(testdataset, batch_size=cfg.batch_size,
                         pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
device = 'cpu'

# Network Setup
# net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)

net = Model(MobileNetV2Dense)
# net.base.migrate(state_dict, force=True, verbose=2)

log_name = 'centerface_logs/dense/training'
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
callbacks = [loss_callback]

if device == 'tpu':
    trainer = pl.Trainer(
        max_epochs=140,
        logger=logger,
        callbacks=callbacks,
        tpu_cores=8
    )
elif device == 'gpu':
    trainer = pl.Trainer(
        max_epochs=140,
        logger=logger,
        callbacks=callbacks,
        gpus=1,
        precision=16
    )
else:
    trainer = pl.Trainer(
        max_epochs=140,
        logger=logger,
        callbacks=callbacks
    )

trainer.fit(net, trainloader, testloader)
