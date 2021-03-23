from models.mobilenetv2.mobilenetv2 import MobileNetV2
import os
import os.path as osp
from time import time

import torch
import numpy as np
from PIL import Image

# local imports
from config import Config as cfg
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2
from models.model import Model
from utils import VisionKit

# device = 'cpu'
# checkpoint_path = 'centerface_logs/tuning/version_1/checkpoints/checkpoint-epoch=77-val_loss=0.0594.ckpt'
# if device == 'cpu':
#     checkpoint = torch.load(
#         checkpoint_path, map_location=lambda storage, loc: storage)
# else:
#     checkpoint = torch.load(checkpoint_path)
# state_dict = checkpoint['state_dict']
# state_dict = checkpoint

net = Model(MobileNetV2VGGBlock)
net.eval()
# net.base.release()


net2 = Model(MobileNetV2)
net2.eval()

with torch.no_grad():
    x = torch.Tensor(16, 3, 416, 416)
    start = time()
    for i in range(100):
        net(x)
    end = time()
    print('tempered: {}'.format(end-start))

with torch.no_grad():
    x = torch.Tensor(16, 3, 416, 416)
    start = time()
    for i in range(100):
        net2(x)
    end = time()
    print('orig: {}'.format(end-start))