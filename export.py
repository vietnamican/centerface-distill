import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from convbnmerge import merge

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetTemperWrapper, configs, MobileNetV2VGGBlockTemper1
from datasets import WiderFace

device = 'cpu'

config = configs[1]
# net = MobileNetTemperWrapper(
#     config['orig'](),
#     config['tempered'](),
#     'tuning',
#     config['orig_module_names'],
#     config['tempered_module_names'],
#     config['is_trains'],
# )
checkpoint_path = 'checkpoint-epoch=93-val_loss=0.0497.ckpt'
if device == 'cpu' or device == 'tpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']

net = Model(MobileNetV2VGGBlockTemper1)
net.eval()
net.migrate(state_dict, force=True, verbose=2)
net.release()
# net.eval()
print(net)


batch_size = 1
height = 480
width = 640

x = torch.randn(batch_size, 3, height, width, requires_grad=True)
output = net(x)

# torch.save(net, 'pt.pt')

torch.onnx.export(net,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  # where to save the model (can be a file or file-like object)
                  "./model.onnx",
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                #   do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  # the model's output names
                  output_names=['hm', 'wh', 'lm', 'off'],
                  dynamic_axes={'input': {0: "batch_size"},  # 2: "height", 3: "width"}, # variable lenght axes
                                'hm': {0: "batch_size"},
                                'wh': {0: "batch_size"},
                                'lm': {0: "batch_size"},
                                'off': {0: "batch_size"}}  # 2: "height", 3: "width"}})
                  )
# net.export('temp.ckpt')
