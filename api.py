import os
import os.path as osp

import torch
import numpy as np
from PIL import Image

# local imports
from config import Config as cfg
from models.mobilenetv2 import MobileNetV2
from models.model import Model
from utils import VisionKit


# def load_model():
#     net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)
#     path = osp.join(cfg.checkpoints, cfg.restore_model)
#     weights = torch.load(path, map_location=cfg.device)
#     net.load_state_dict(weights)
#     net.eval()
#     return net

device = 'cpu'
# checkpoint_path = 'centerface_logs/training/version_0/checkpoints/checkpoint-epoch=66-val_loss=0.0583.ckpt'
checkpoint_path = 'checkpoints/final.pth'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
# state_dict = checkpoint['state_dict']
state_dict = checkpoint

net = Model(MobileNetV2)
net.eval()
net.migrate(state_dict, force=True, verbose=2)

def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep

def preprocess(im):
    new_im, _, _, *params = VisionKit.letterbox(im, cfg.insize)
    return new_im, params

def postprocess(bboxes, landmarks, params):
    bboxes, landmarks = VisionKit.letterbox_inverse(*params, bboxes, landmarks, skip=2)
    return bboxes, landmarks

def detect(im):
    data = cfg.test_transforms(im)
    data = data[None, ...]
    with torch.no_grad():
        out = net(data)
    return out[0]

def decode(out):
    hm = out['hm']
    wh = out['wh']
    off = out['off']
    lm = out['lm']
    hm = VisionKit.nms(hm, kernel=3)
    hm.squeeze_()
    off.squeeze_()
    wh.squeeze_()
    lm.squeeze_()

    hm = hm.numpy()
    hm[hm < cfg.threshold] = 0
    xs, ys = np.nonzero(hm)
    bboxes = []
    scores = []
    landmarks = []
    for x, y in zip(xs, ys):
        ow = off[0][x, y]
        oh = off[1][x, y]
        cx = (ow + y) * 4
        cy = (oh + x) * 4 

        w = wh[0][x, y]
        h = wh[1][x, y]
        width = np.exp(w) * 4
        height = np.exp(h) * 4
        
        left = cx - width / 2
        top = cy - height / 2
        right = cx + width / 2
        bottom = cy + height / 2
        bboxes.append([left, top, right, bottom])
        scores.append(hm[x, y])

        # landmark
        lms = []
        for i in range(0, 10, 2):
            lm_x = lm[i][x, y]
            lm_y = lm[i+1][x, y]
            lm_x = lm_x * width + left 
            lm_y = lm_y * height + top 
            lms += [lm_x, lm_y]
        landmarks.append(lms)
    bboxes = np.array(bboxes)
    landmarks = np.array(landmarks)
    keep_indexes = nms(bboxes, scores, 0.4)
    return bboxes[keep_indexes], landmarks[keep_indexes]
    return bboxes, landmarks


def visualize(im, bboxes, landmarks):
    return VisionKit.visualize(im, bboxes, landmarks, skip=2)


if __name__ == '__main__':
    impath = 'samples/c.jpg'
    im = Image.open(impath)
    new_im, params = preprocess(im)
    pred = detect(new_im)
    bboxes, landmarks = decode(pred)
    print(len(bboxes))
    bboxes, landmarks = postprocess(bboxes, landmarks, params)
    visualize(im, bboxes, landmarks)