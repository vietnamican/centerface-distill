import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl

from .mnet import CenterFace
from .loss import RegLoss


class Model(CenterFace):
    def __init__(self, head_conv=24, pretrained=True):
        super(Model, self).__init__()
        self.heatmap_loss = nn.MSELoss()
        self.wh_loss = RegLoss()
        self.off_loss = RegLoss()
        self.lm_loss = RegLoss()
        self.lambdas = [1, 1, 0.1, 0.1]

    def shared_step(self, batch):
        data, labels = batch
        out = self.forward(data)
        heatmaps, offs, whs, lms = out[0], out[1], out[2], out[3]

        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0:1])
        l_off = self.off_loss(offs, labels[:, [1, 2]])
        l_wh = self.wh_loss(whs, labels[:, [3, 4]])
        l_lm = self.lm_loss(lms, labels[:, 5:])

        return l_heatmap, l_off, l_wh, l_lm

    def training_step(self, batch, batch_idx):
        l_heatmap, l_off, l_wh, l_lm = self.shared_step(batch)
        loss = self.lambdas[0] * l_heatmap + self.lambdas[1] * \
            l_off + self.lambdas[2] * l_wh + self.lambdas[3] * l_lm
        self.log_dict({'loss': loss, 'l_hm': l_heatmap,
                      'l_off': l_off, 'l_wh': l_wh, 'l_lm': l_lm}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        l_heatmap, l_off, l_wh, l_lm = self.shared_step(batch)
        loss = self.lambdas[0] * l_heatmap + self.lambdas[1] * \
            l_off + self.lambdas[2] * l_wh + self.lambdas[3] * l_lm
        self.log_dict({'val_loss': loss, 'val_l_hm': l_heatmap,
                      'val_l_off': l_off, 'val_l_wh': l_wh, 'val_l_lm': l_lm}, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [90, 120], 0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
