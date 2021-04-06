import os
import os.path as osp
from time import time

import torch
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetV2
from datasets import WiderFace, WiderFaceVal

device = 'cpu'
checkpoint_path = 'checkpoint-epoch=139-val_loss=0.0515.ckpt'
# checkpoint_path = 'checkpoints/final.pth'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
# state_dict = checkpoint

net = Model(MobileNetV2VGGBlockTemper1)
# net = Model(MobileNetV2)
net.eval()
net.migrate(state_dict, force=True, verbose=2)
# net.release()
for i, (name, p) in enumerate(net.named_parameters()):
    print(i, name)

net.eval()
x = torch.Tensor(1, 3, 1024, 1024)
with torch.no_grad():
    start = time()
    for i in range(100):
        net(x)
    stop = time()
    print(stop-start)