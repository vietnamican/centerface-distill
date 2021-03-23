import torch
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
import torch.utils.model_zoo as model_zoo

from models.model import Model
from datasets import WiderFaceVal
from config import Config as cfg

traindataset = WiderFaceVal(cfg.valdataroot, cfg.valannfile, cfg.sigma,
                         cfg.downscale, cfg.insize, cfg.train_transforms)
trainloader = DataLoader(traindataset, batch_size=cfg.batch_size,
                         pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

for im, hm in trainloader:
    print(im.shape)
    print(hm.shape)
