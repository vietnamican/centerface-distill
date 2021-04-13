import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


def _neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class RegLoss(nn.Module):
    """Regression loss for CenterFace, especially 
    for offset, size and landmarks
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt):
        mask = gt.gt(0)
        pred = pred[mask]
        gt = gt[mask]
        loss = self.loss(pred, gt)
        loss = loss / (mask.float().sum() + 1e-4)
        return loss


class NegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = _neg_loss

    def forward(self, pred, gt):
        return self.loss(pred, gt)


class AverageMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, correct, total):
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct.float() / self.total
