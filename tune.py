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
from models.mobilenetv2 import MobileNetV2VGGBlock
from datasets import WiderFace

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                progress=True)

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

# Network Setup
# net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)

checkpoint_path = 'checkpoints/final.pth'
if device == 'cpu' or device == 'tpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)

net = Model(MobileNetV2VGGBlock)
net.migrate(net.dla_up.state_dict(), net.filter_state_dict_with_prefix(checkpoint, 'dla_up'), force=True, verbose=1)
net.migrate(net.hm.state_dict(), net.filter_state_dict_with_prefix(checkpoint, 'hm'), force=True, verbose=1)
net.migrate(net.wh.state_dict(), net.filter_state_dict_with_prefix(checkpoint, 'wh'), force=True, verbose=1)
net.migrate(net.off.state_dict(), net.filter_state_dict_with_prefix(checkpoint, 'off'), force=True, verbose=1)
net.migrate(net.lm.state_dict(), net.filter_state_dict_with_prefix(checkpoint, 'lm'), force=True, verbose=1)

checkpoint_path = 'config0_checkpoint-epoch=53-val_loss=2.8531.ckpt'
if device == 'cpu' or device == 'tpu':
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
net.base.migrate(checkpoint, force=True, verbose=1)


log_name = 'centerface_logs/tuning'
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
        max_epochs=90,
        logger=logger,
        callbacks=callbacks,
        tpu_cores=8
    )
elif device == 'gpu':
    trainer = pl.Trainer(
        max_epochs=90,
        logger=logger,
        callbacks=callbacks,
        gpus=1
    )
else:
    trainer = pl.Trainer(
        max_epochs=90,
        logger=logger,
        callbacks=callbacks
    )

trainer.fit(net, trainloader, valloader)
