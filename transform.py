from models.mobilenetv2.mobilenetv2 import MobileNetV2Dense
import os
import os.path as osp
from time import time
import numpy as np

import torch
from torch.utils.data import DataLoader
import cv2
from PIL import Image

from config import Config as cfg
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1, MobileNetV2, MobileNetV2Dense
from datasets import WiderFace, WiderFaceVal
from api import preprocess, decode, postprocess, detect, visualize

def load_model():
    device = 'cpu'
    # checkpoint_path = 'checkpoint-epoch=42-val_loss=0.0498.ckpt'
    checkpoint_path = 'checkpoint-epoch=93-val_loss=0.0497.ckpt'
    # checkpoint_path = 'checkpoints/final.pth'
    # checkpoint_path = 'checkpoints/centerface.pt'
    if device == 'cpu':
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    
    # net = Model(MobileNetV2)
    # net = Model(MobileNetV2Dense)
    net = Model(MobileNetV2VGGBlockTemper1)
    # net = checkpoint
    net.eval()
    net.migrate(state_dict, force=True, verbose=1)
    net.release()
    net.eval()

    return net

if __name__ == '__main__':
    net = load_model()
    print(net)
    # print(net.dla_up.conv_last.cbr.conv.bias)
    # print(net.dla_up.up_2.up.cbr.weight)
    # print(net.dla_up.up_2.up[1].bias)
    # print(net.dla_up.up_2.up[1].running_mean)
    # image_orig = cv2.imread('000388.jpg')
    image_orig = cv2.imread('data/WIDER_val/images/43--Row_Boat/43_Row_Boat_Canoe_43_81.jpg')
    # print(image[:,:,0])
    image = Image.fromarray(image_orig)
    new_im, params = preprocess(image)
    data = cfg.test_transforms(new_im)
    data = data[None, ...]
    with torch.no_grad():
        out = net(data)
    # hm = out[0].squeeze_()
    # print(hm.shape)
    # print((hm >0.5).sum())
    # for line in hm:
    #     print(line)



    # conv2_out = net.base.feature_1[0].forward_path(data)
    # conv2_out = net.base.feature_1[0].relu(conv2_out)
    # conv2_out = net.base.feature_1[1].forward_path(conv2_out)
    # print(conv2_out.shape)
    # print(conv2_out[0, 2, 0])
    # # for i in range(70, 80):
    # #     print(data[0,0,i])
    # with torch.no_grad():
    #     out = net(data)
    # heatmap = out[0].squeeze_()
    # print(heatmap[0])

    bboxes, landmarks = decode(out)
    new_im = np.array(new_im)
    if bboxes.shape[0] > 0:
        # print("has {} face".format(bboxes.shape[0]))
        bboxes, landmarks = postprocess(bboxes, landmarks, params)
        for bbox in bboxes:
            left, top, right, bottom = map(int, bbox)
            cv2.rectangle(image_orig, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
    
    cv2.imshow('frame', image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # imname = 'data/WIDER_val/images/43--Row_Boat/43_Row_Boat_Canoe_43_81.jpg'
    # imname = '000388.jpg'
    # # idx = 6
    # # imname = '0--Parade/0_Parade_Parade_0_470.jpg'
    # # impath = osp.join('data', 'WIDER_val', 'images', imname)
    # # imname = imname.split('/')[-1]
    # impath = imname

    # im = Image.open(impath)
    # new_im, params = preprocess(im)
    # pred = detect(net, new_im)
    # bboxes, landmarks = decode(pred)
    # bboxes, landmarks = postprocess(bboxes, landmarks, params)
    # # print(result, len(bboxes))
    # # print(result, len(gt_bboxes))
    # print("aaaaaaaaaa")
    # visualize(im, bboxes, landmarks)