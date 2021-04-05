import os
import os.path as osp
from time import time

import torch
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary
from convbnmerge import merge

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetV2, MobileNetV2Dense
from datasets import WiderFace, WiderFaceVal


torch.set_grad_enabled(False)

device = 'cpu'

# net = Model(MobileNetV2VGGBlockTemper1)
net = Model(MobileNetV2Dense)
# net = Model(MobileNetV2)
# net.migrate(state_dict, force=True, verbose=2)
net.release()
# net.eval()

batch_size = 1
height = 480
width = 640

# x = torch.randn(batch_size, 3, height, width, requires_grad=True)
# output = net(x)

# for i, (name, p) in enumerate(net.named_parameters()):
#     print(i, name)


summary(net, (3, 32*6, 32*6), col_names=['mult_adds', 'num_params', 'kernel_size'], depth=6)
# summary(net, x)
merge(net)
x = torch.Tensor(1, 3, 32*6, 32*6)
with torch.no_grad():
    start = time()
    for i in range(100):
        net(x)
    stop = time()
print((stop-start)/100)

net = Model(MobileNetV2)
net.eval()
merge(net)
with torch.no_grad():
    start = time()
    for i in range(100):
        net(x)
    stop = time()
print((stop-start)/100)
