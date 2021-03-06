from models.tempered_model import TemperedModel
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from ..tempered_model import TemperedModel


class MobileNetTemperWrapper(TemperedModel):

    def __init__(self, orig, tempered, mode, orig_module_names, tempered_module_names, is_trains, prun_module=None):
        super().__init__(orig, tempered, mode, orig_module_names,
                         tempered_module_names, is_trains, prun_module)

    def configure_optimizers(self):
        if self.mode == 'temper':
            params = []
            for tempered_modules in self.tempered_modules:
                if isinstance(tempered_modules, list):
                    for tempered_module in tempered_modules:
                        params.extend(tempered_module.parameters())
                else:
                    params.extend(tempered_modules.parameters())
            optimizer = torch.optim.SGD(params, lr=0.001,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=70)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
