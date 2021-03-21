import os
import os.path as osp
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

from config import Config as cfg
from models.loss import RegLoss
from models.mnet import CenterFace
from datasets import WiderFace
from models.mobilenetv2 import MobileNetV2, MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetTemperWrapper
from models.model import Model
from models.mobilenetv2 import configs
# model = CenterFace()

# for name, c in model.base.named_children():
#     print(name)

# checkpoint_path = 'checkpoints/final.pth'
# checkpoint = torch.load(checkpoint_path, map_location= lambda storage, loc: storage)
# model.migrate(checkpoint, force=True, verbose=2)
# load_model(model, checkpoint)

# mbnet = MobileNetV2()
# mbnet = MobileNetV2VGGBlock()
# mbnet = MobileNetV2VGGBlockTemper1()
# mbnet.release()
# x = torch.Tensor(1, 3, 416, 416)
# start = time()
# for i in range(100):
#     mbnet(x)
# stop = time()
# print(stop - start)
# summary(
#     mbnet,
#     (3, 416, 416),
#     # col_names=["input_size", "output_size", "num_params", "kernel_size"],
#     depth=5
# )
config = configs[0]
device = 'cpu'
mode = 'tuning'
model = MobileNetTemperWrapper(
    config['orig'](), config['tempered'](), mode, config["orig_module_names"], config["tempered_module_names"], config["is_trains"])
checkpoint_path = 'centerface_logs/temper/version_0/checkpoints/checkpoint-epoch=23-val_loss=3.3087.ckpt'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
model.migrate(state_dict, force=True, verbose=2)
model.export('mobilenetv2_vggblock.ckpt')
print(model.forward_path)