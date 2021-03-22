from torchsummary import summary
import torch.utils.model_zoo as model_zoo

from models.model import Model
from models.mnet import MobileNetV2


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                              progress=True)

net = Model(MobileNetV2)
net.base.migrate(state_dict, force=True, verbose=2)

summary(net, (3, 416, 416))
