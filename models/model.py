import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl

from .mnet import MobileNetSeg
from .base import Base
from .loss import RegLoss


class Model(MobileNetSeg):
    def __init__(self, base):
        super().__init__(base, {'hm':1, 'wh':2, 'lm':10, 'off':2})
        self.heatmap_loss = nn.MSELoss()
        self.wh_loss = RegLoss()
        self.off_loss = RegLoss()
        self.lm_loss = RegLoss()


    def training_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = self.off_loss(offs, labels[:, [1,2]])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [3,4]])
        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = self.lm_loss(lms, labels[:, 5:])

        self.log_dict({'t_heat': l_heatmap, 't_off': l_off, 't_size': l_wh, 't_landmark': l_lm}, prog_bar=True)
        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        self.log_dict({'train_loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = self.off_loss(offs, labels[:, [1,2]])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [3,4]])
        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = self.lm_loss(lms, labels[:, 5:])

        self.log_dict({'v_heat': l_heatmap, 'v_off': l_off, 'v_size': l_wh, 'v_landmark': l_lm}, prog_bar=False)
        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        self.log_dict({'val_loss': loss})
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer