import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from models.loss import RegLoss
from models.mnet import CenterFace
from datasets import WiderFace

model = CenterFace()

for name, c in model.base.named_children():
    print(name)

checkpoint_path = 'checkpoints/final.pth'
checkpoint = torch.load(checkpoint_path, map_location= lambda storage, loc: storage)
model.migrate(checkpoint, force=True, verbose=2)
# load_model(model, checkpoint)