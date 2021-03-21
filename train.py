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

from config import Config as cfg
from models.loss import RegLoss
from models.model import Model
from datasets import WiderFace

try:
    import torch_xla.core.xla_model as xm
except:
    pass

pl.seed_everything(42)

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

device = 'cpu'
if device == 'tpu':
    traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                             cfg.downscale, cfg.insize, cfg.train_transforms, 'train')
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        traindataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    trainloader = DataLoader(traindataset, sampler=trainsampler, batch_size=cfg.batch_size,
                             pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

    valdataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                           cfg.downscale, cfg.insize, cfg.train_transforms, 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(
        valdataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    valloader = DataLoader(valdataset, sampler=valsampler, batch_size=cfg.batch_size,
                           pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
else:
    # Data Setup
    traindataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                             cfg.downscale, cfg.insize, cfg.train_transforms, 'train')
    trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                             pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
    valdataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                           cfg.downscale, cfg.insize, cfg.train_transforms, 'val')
    valloader = DataLoader(valdataset, batch_size=cfg.batch_size,
                           pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
# device = cfg.device

# Network Setup
net = Model(base=MobileNetV2VGGBlock)
# state_dict = model_zoo.load_url(model_urls['mobilenet_v2'], progress=True)
# net.base.migrate(state_dict, force=True)
# checkpoint_path = 'checkpoints/final.pth'
checkpoint_path = 'centerface_logs/training/version_0/checkpoints/checkpoint-epoch=66-val_loss=0.0583.ckpt'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']

net.migrate(net.hm.state_dict(), net.filter_state_dict_with_prefix(state_dict, 'hm'), force=True, verbose=2)
net.migrate(net.wh.state_dict(), net.filter_state_dict_with_prefix(state_dict, 'wh'), force=True, verbose=2)
net.migrate(net.lm.state_dict(), net.filter_state_dict_with_prefix(state_dict, 'lm'), force=True, verbose=2)
net.migrate(net.off.state_dict(), net.filter_state_dict_with_prefix(state_dict, 'off'), force=True, verbose=2)

checkpoint_path = 'mobilenetv2_vggblock.ckpt'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
# state_dict = checkpoint['state_dict']
net.base.migrate(checkpoint, force=True, verbose=2)

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
lr_monitor = LearningRateMonitor(logging_interval='epoch')
callbacks = [loss_callback, lr_monitor]

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
# trainer.test(net, trainloader)
