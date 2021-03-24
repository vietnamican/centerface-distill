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
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2
from datasets import WiderFace, WiderFaceVal

testdataset = WiderFaceVal(cfg.valdataroot, cfg.valannfile, cfg.sigma,
                           cfg.downscale, cfg.insize, cfg.train_transforms)
testloader = DataLoader(testdataset, batch_size=4,
                        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
device = 'cpu'

# checkpoint_path = 'checkpoints/final.pth'
checkpoint_path = 'centerface_logs/training/version_0/checkpoints/checkpoint-epoch=89-val_loss=0.0586.ckpt'
if device == 'cpu' or device == 'tpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
# state_dict = checkpoint

net = Model(MobileNetV2)
net.eval()
net.migrate(state_dict, force=True, verbose=2)

log_name = 'centerface_logs/val'
logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    name=log_name,
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
        callbacks=callbacks,
        limit_test_batches=0.1
    )

trainer.test(net, testloader)
