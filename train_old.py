import os
import os.path as osp
from sys import version

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from models.loss import RegLoss
from models.model import Model
from models.mnet import MobileNetV2
from datasets import WiderFace

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                progress=True)


# Data Setup
dataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma, cfg.downscale, cfg.insize, cfg.train_transforms)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, 
    pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
device = cfg.device

# Network Setup
net = Model(base=MobileNetV2)
net.base.migrate(state_dict, force=True, verbose=2)

# Training Setup
optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
heatmap_loss = nn.MSELoss()
wh_loss = RegLoss()
off_loss = RegLoss()
lm_loss = RegLoss()

# Checkpoints Setup
checkpoints = cfg.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if cfg.restore:
    weights_path = osp.join(checkpoints, cfg.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"load weights from checkpoints: {cfg.restore_model}")

# Start training
net.train()
net.to(device)

# Epoch 0/90, heat: 0.009534, off: 0.066591, size: 1.663676, landmark: 0.007572

for e in range(cfg.epoch):
    t = tqdm(dataloader)
    # for data, labels in tqdm(dataloader, desc=f"Epoch {e}/{cfg.epoch}",
    #                          ascii=True, total=len(dataloader)):
                             
    for data, labels in t:
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = net(data)

        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = heatmap_loss(heatmaps, labels[:, 0])

        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = off_loss(offs, labels[:, [1,2]])

        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = wh_loss(whs, labels[:, [3,4]])

        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = lm_loss(lms, labels[:, 5:])

        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        loss.backward()
        optimizer.step()
        t.set_description(f"Epoch {e}/{cfg.epoch}, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}, landmark: {l_lm:.6f}")

    print(f"Epoch {e}/{cfg.epoch}, heat: {l_heatmap:.6f}, off: {l_off:.6f}, size: {l_wh:.6f}, landmark: {l_lm:.6f}")

    backbone_path = osp.join(checkpoints, f"{e}.pth")
    torch.save(net.state_dict(), backbone_path)

