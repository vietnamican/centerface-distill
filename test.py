import torch
from torchsummary import summary
import torch.utils.model_zoo as model_zoo

from models.model import Model
from models.mnet import MobileNetV2


# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }

# state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
#                                               progress=True)
                                            
device = 'cpu'
checkpoint_path = 'centerface_logs/training/version_0/checkpoints/checkpoint-epoch=66-val_loss=0.0583.ckpt'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
# state_dict = checkpoint['state_dict']
state_dict = checkpoint

net = Model(MobileNetV2)
net.migrate(state_dict, force=True, verbose=2)

summary(net, (3, 416, 416))
