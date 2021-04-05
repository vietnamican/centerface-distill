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

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetV2, MobileNetV2Dense
from datasets import WiderFace, WiderFaceVal


torch.set_grad_enabled(False)

device = 'cpu'
# checkpoint_path = 'centerface_logs/tuning_vgg_vggblocktemper1/version_2/checkpoints/checkpoint-epoch=139-val_loss=0.0515.ckpt'
checkpoint_path = 'checkpoints/final.pth'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
# state_dict = checkpoint['state_dict']
state_dict = checkpoint

# net = Model(MobileNetV2VGGBlockTemper1)
net = Model(MobileNetV2Dense)
# net = Model(MobileNetV2)
# net.migrate(state_dict, force=True, verbose=2)
# net.release()
# net.eval()

batch_size = 1
height = 480
width = 640

# x = torch.randn(batch_size, 3, height, width, requires_grad=True)
# output = net(x)

# for i, (name, p) in enumerate(net.named_parameters()):
#     print(i, name)

x = torch.Tensor(1, 3, 32*6, 32*6)
# summary(net, x, col_names=['mult_adds', 'num_params', 'kernel_size'], depth=6)
# summary(net, x)
with torch.no_grad():
    start = time()
    for i in range(100):
        net(x)
    stop = time()
print((stop-start)/100)


net = Model(MobileNetV2)
net.eval()
with torch.no_grad():
    start = time()
    for i in range(100):
        net(x)
    stop = time()
print((stop-start)/100)