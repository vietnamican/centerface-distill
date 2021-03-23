import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import numpy as np

from .mnet import MobileNetSeg
from .base import Base
from .loss import RegLoss


class Model(MobileNetSeg):
    def __init__(self, base):
        super().__init__(base, {'hm': 1, 'wh': 2, 'lm': 10, 'off': 2})
        self.threshold = 0.4
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
        l_off = self.off_loss(offs, labels[:, [1, 2]])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [3, 4]])
        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = self.lm_loss(lms, labels[:, 5:])

        self.log_dict({'t_heat': l_heatmap, 't_off': l_off,
                       't_size': l_wh, 't_landmark': l_lm}, prog_bar=True)
        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        self.log_dict({'train_loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = self.off_loss(offs, labels[:, [1, 2]])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [3, 4]])
        lms = torch.cat([o['lm'].squeeze() for o in out], dim=0)
        l_lm = self.lm_loss(lms, labels[:, 5:])

        self.log_dict({'v_heat': l_heatmap, 'v_off': l_off,
                       'v_size': l_wh, 'v_landmark': l_lm}, prog_bar=False)
        loss = l_heatmap + l_off + l_wh * 0.1 + l_lm * 0.1
        self.log_dict({'val_loss': loss})

        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        heatmaps = torch.cat([o['hm'].squeeze() for o in out], dim=0)
        l_heatmap = self.heatmap_loss(heatmaps, labels[:, 0])
        offs = torch.cat([o['off'].squeeze() for o in out], dim=0)
        l_off = self.off_loss(offs, labels[:, [1, 2]])
        whs = torch.cat([o['wh'].squeeze() for o in out], dim=0)
        l_wh = self.wh_loss(whs, labels[:, [3, 4]])
        self.log_dict({'heat': l_heatmap, 'off': l_off,
                       'size': l_wh}, prog_bar=False)
        pred_bboxes = self._decode(out)
        gt_bboxes = self._decode(
            {'hm': labels[:, 0], 'off': labels[: [1, 2]], 'wh': labels[:, [3, 4]]})
        Model._caculate_accuracy(pred_bboxes, gt_bboxes)

    @staticmethod
    def _is_overlap(pred_box, gt_box, threshold=0.4):
        x1min, y1min, x1max, y1max = pred_box
        x2min, y2min, x2max, y2max = gt_box
        overlap = max(min(x1max, x2max) - max(x1min, x2min) + 1, 0) * max(min(y1max, y2max) - max(y1min, y2min) + 1, 0)
        overall = (x1max - x1min + 1) * (y1max - y1min + 1) + (x2max - x2min + 1) * (y2max - y2min + 1) - overlap
        if overlap / overall >= threshold:
            return True
        return False

    @staticmethod
    def _calculate_accuracy(pred_bboxes, gt_bboxes):
        pred_centers = [[(left+right)/2, (top+bottom)/2]
                        for left, top, right, bottom in pred_bboxes]
        gt_centers = [[(left+right)/2, (top+bottom)/2]
                      for left, top, right, bottom in gt_bboxes]

        def _manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        result = 0
        for i, pred_center in enumerate(pred_centers):
            closest_index = -1
            min_distance = 2000
            for j, gt_center in enumerate(gt_centers):
                if _manhattan_distance(pred_center, gt_center) < min_distance:
                    closest_index = j
            if Model._is_overlap(pred_bboxes[i], gt_bboxes[closest_index]):
                result += 1

        precision = result / len(pred_bboxes)
        recall = result / len(gt_bboxes)
        return precision, recall

    @staticmethod
    def _nms(heat, kernel):
        padding = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heat, kernel, stride=1, padding=padding)
        keep = (hmax == heat).float()
        return heat * keep

    def _decode(self, out):
        hm = out['hm']
        wh = out['wh']
        off = out['off']
        hm = Model._nms(hm, kernel=3)
        hm.squeeze_()
        off.squeeze_()
        wh.squeeze_()

        hm = hm.numpy()
        hm[hm < self.threshold] = 0
        xs, ys = np.nonzero(hm)
        bboxes = []
        for x, y in zip(xs, ys):
            ow = off[0][x, y]
            oh = off[1][x, y]
            cx = (ow + y) * 4
            cy = (oh + x) * 4

            w = wh[0][x, y]
            h = wh[1][x, y]
            width = np.exp(w) * 4
            height = np.exp(h) * 4

            left = cx - width / 2
            top = cy - height / 2
            right = cx + width / 2
            bottom = cy + height / 2
            bboxes.append([left, top, right, bottom])

        return bboxes

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
