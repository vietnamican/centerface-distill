import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pred.unsqueeze_(1)
    pred = F.sigmoid(pred)
    gt.unsqueeze_(1)
    # print(pred.shape)
    # print(gt.shape)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # print(neg_weights)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
        neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
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
        # print(correct)
        # print(total)
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct.float() / self.total
