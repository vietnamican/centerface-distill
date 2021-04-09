from functools import partial
from models.tempered_model import TemperedModel
import torch
from torch import nn
import pytorch_lightning as pl

from .tempered_model import TemperedModel
from .mnet import MobileNetSeg
from .base import ConvBatchNormRelu, ConvTransposeBatchNormRelu
from .loss import RegLoss


class CenterNetUpTemper(nn.Module):
    def __init__(self, mode='temper'):
        super().__init__()
        self.mode = mode
        self.conv = ConvBatchNormRelu(
            320, 24, kernel_size=3, padding=1, bias=False)
        self.conv_last = ConvBatchNormRelu(
            24, 24, kernel_size=3, padding=1, bias=False)
        self.up_0 = ConvTransposeBatchNormRelu(
            24, 24, kernel_size=2, stride=2, groups=24, bias=False)
        self.up_1 = ConvTransposeBatchNormRelu(
            24, 24, kernel_size=2, stride=2, groups=24, bias=False)
        self.up_2 = ConvTransposeBatchNormRelu(
            24, 24, kernel_size=2, stride=2, groups=24, bias=False)
        self._register_forward()

    def _register_forward(self):
        if self.mode == 'temper':
            def _forward(self, x):
                x1 = self.conv(x)
                x2 = self.up_0(x1)
                x3 = self.up_1(x2)
                x4 = self.up_2(x3)
                x5 = self.conv_last(x4)
                return x1, x2, x3, x4, x5
        elif self.mode == 'tuning':
            def _forward(self, x):
                return self.conv_last(self.up_2(self.up_1(self.up_0(self.conv(x)))))
        self._forward = partial(_forward, self)

    def forward(self, x):
        return self._forward(x)


class CenternetTemperWrapper(MobileNetSeg):

    def __init__(self, base, heads, head_conv=24, pretrained=True, mode='temper'):
        super().__init__(base, heads, head_conv=24, pretrained=True)
        self.mode = mode
        self.dla_up_temper = CenterNetUpTemper(mode)
        self._register_forward()
        self.criterion = nn.MSELoss()
        self.heatmap_loss = nn.MSELoss()
        self.wh_loss = RegLoss()
        self.off_loss = RegLoss()
        self.lm_loss = RegLoss()

    def _register_forward(self):
        if self.mode == 'temper':
            def _forward(self, x):
                x = self.base(x)
                _x1, _x2, _x3, _x4, _x5 = self.dla_up_temper(x[-1])
                layers = list(x)
                x1 = self.dla_up.conv(layers[-1])
                x2 = self.dla_up.up_0([x1, layers[-2]])
                x3 = self.dla_up.up_1([x2, layers[-3]])
                x4 = self.dla_up.up_2([x3, layers[-4]])
                x5 = self.dla_up.conv_last(x4)
                return _x1, _x2, _x3, _x4, _x5, x1, x2, x3, x4, x5
        elif self.mode == 'tuning':
            def _forward(self, x):
                x = self.base(x)
                # x = self.dla_up(x)
                x = self.dla_up_temper(x[-1])
                ret = []
                for head in self.heads:
                    ret.append(self.__getattr__(head)(x))
                return ret

        self._forward = partial(_forward, self)

    def forward(self, x):
        return self._forward(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        if self.mode == 'temper':
            _x1, _x2, _x3, _x4, _x5, x1, x2, x3, x4, x5 = self(data)
            loss1 = self.criterion(_x1, x1)
            loss2 = self.criterion(_x2, x2)
            loss3 = self.criterion(_x3, x3)
            loss4 = self.criterion(_x4, x4)
            loss5 = self.criterion(_x5, x5)
            self.log('train_loss_conv', loss1)
            self.log('train_loss_1', loss2)
            self.log('train_loss_2', loss3)
            self.log('train_loss_3', loss4)
            self.log('train_loss_last', loss5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            self.log('train_loss', loss)
            return loss
        else:
            out = self(data)
            heatmaps = out[0]
            l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0:1])
            offs = out[3]
            l_off = self.off_loss(offs, labels[:, [1, 2]])
            whs = out[1]
            l_wh = self.wh_loss(whs, labels[:, [3, 4]])
            lms = out[2]
            l_lm = self.lm_loss(lms, labels[:, 5:])

            self.log_dict({'t_heat': l_heatmap, 't_off': l_off,
                           't_size': l_wh, 't_landmark': l_lm}, prog_bar=True)
            loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
            self.log_dict({'train_loss': loss})
            return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        if self.mode == 'temper':
            _x1, _x2, _x3, _x4, _x5, x1, x2, x3, x4, x5 = self(data)
            loss1 = self.criterion(_x1, x1)
            loss2 = self.criterion(_x2, x2)
            loss3 = self.criterion(_x3, x3)
            loss4 = self.criterion(_x4, x4)
            loss5 = self.criterion(_x5, x5)
            self.log('val_loss_conv', loss1)
            self.log('val_loss_1', loss2)
            self.log('val_loss_2', loss3)
            self.log('val_loss_3', loss4)
            self.log('val_loss_last', loss5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            self.log('val_loss', loss)
            return loss
        else:
            out = self(data)
            heatmaps = out[0]
            l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0:1])
            offs = out[3]
            l_off = self.off_loss(offs, labels[:, [1, 2]])
            whs = out[1]
            l_wh = self.wh_loss(whs, labels[:, [3, 4]])
            lms = out[2]
            l_lm = self.lm_loss(lms, labels[:, 5:])

            self.log_dict({'v_heat': l_heatmap, 'v_off': l_off,
                           'v_size': l_wh, 'v_landmark': l_lm}, prog_bar=False)
            loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
            self.log_dict({'val_loss': loss})
            return loss

    def configure_optimizers(self):
        if self.mode == 'temper':
            self.freeze_except_prefix('dla_up_temper')
            optimizer = torch.optim.Adam(self.dla_up_temper.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
