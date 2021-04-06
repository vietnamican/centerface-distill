import os
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# local imports
from config import Config as cfg
from models.mobilenetv2 import MobileNetV2, MobileNetV2VGGBlockTemper1, MobileNetV2Dense
from models.model import Model
from utils import VisionKit
from datasets import WiderFace
from models.loss import AverageMetric

testdataset = WiderFace(cfg.valdataroot, cfg.valannfile, cfg.sigma,
                           cfg.downscale, cfg.insize, cfg.train_transforms)
testloader = DataLoader(testdataset, batch_size=1,
                        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
precision = AverageMetric()
recall = AverageMetric()
# def load_model():
#     net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)
#     path = osp.join(cfg.checkpoints, cfg.restore_model)
#     weights = torch.load(path, map_location=cfg.device)
#     net.load_state_dict(weights)
#     net.eval()
#     return net

device = 'cpu'
checkpoint_path = 'checkpoint-epoch=05-val_loss=0.0656.ckpt'
# checkpoint_path = 'checkpoints/final.pth'
if device == 'cpu':
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
# state_dict = checkpoint

net = Model(MobileNetV2VGGBlockTemper1)
net.eval()
net.migrate(state_dict, force=True, verbose=1)
net.release()


def iou(box1, box2):
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[2]
    y21 = box1[3]

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[2]
    y22 = box2[3]

    xx1 = max(x11, x12)
    yy1 = max(y11, y12)
    xx2 = min(x21, x22)
    yy2 = min(y21, y22)

    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)

    overlap = w * h
    overall = (x21 - x11 + 1) * (y21 - y11 + 1) + \
        (x22 - x12 + 1) * (y22 - y12 + 1) - overlap
    return overlap / 1.0 / overall


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


def postprocess(bboxes, params):
    bboxes, _ = VisionKit.letterbox_inverse(
        *params, bboxes, skip=2)
    return bboxes


def detect(net, im):
    data = cfg.test_transforms(im)
    data = data[None, ...]
    with torch.no_grad():
        out = net(data)
    return out[0]


def decode(out):
    hm = out['hm']
    wh = out['wh']
    off = out['off']
    hm = VisionKit.nms(hm, kernel=3)
    hm.squeeze_()
    off.squeeze_()
    wh.squeeze_()

    hm = hm.numpy()
    hm[hm < cfg.threshold] = 0
    xs, ys = np.nonzero(hm)
    bboxes = []
    scores = []
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

        
    bboxes = np.array(bboxes)
    if len(bboxes) == 0:
        return bboxes
    keep_indexes = nms(bboxes, scores, 0.4)
    return bboxes[keep_indexes]


def visualize(im, bboxes, name):
    return VisionKit.visualize(im, bboxes, skip=2, name=name)


def manhattan_distance(box1, box2):
    center1 = (box1[2] - box1[0], box1[3] - box1[1])
    center2 = (box2[2] - box2[0], box2[3] - box2[1])
    return abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])

def calculate_metrics(pred_bboxes, gt_bboxes):
    iou_threshold = 0.4
    is_occupied = [False]*gt_bboxes.shape[0]
    pred_index = []
    gt_index = []
    result = 0
    for i, pred_box in enumerate(pred_bboxes):
        max_iou = 0
        for j, gt_box in enumerate(gt_bboxes):

            if not is_occupied[j]:
                iou_distance = iou(pred_box, gt_box) 
                if iou_distance > max_iou:
                    closest_index = j
                    max_iou = iou_distance
        if max_iou > iou_threshold:
            result += 1
            is_occupied[closest_index] = True
            pred_index.append(i)
            gt_index.append(closest_index)
    return result, pred_index, gt_index

if not os.path.isdir('resultdense'):
    os.mkdir('resultdense')

if __name__ == '__main__':
    # i = 0
    for im, labels, imname, idx in tqdm(testloader):
        try:
            # i += 1
            # if i == 10:
            #     break
            imname = imname[0]
            # idx = 6
            # imname = '0--Parade/0_Parade_Parade_0_470.jpg'
            impath = osp.join('data', 'WIDER_val', 'images', imname)
            imname = imname.split('/')[-1]
            gt_bboxes = np.array(testdataset.annslist[idx])
            gt_bboxes[:, 2] += gt_bboxes[:, 0]
            gt_bboxes[:, 3] += gt_bboxes[:, 1]

            im = Image.open(impath)
            new_im, params = preprocess(im)
            pred = detect(net, new_im)
            bboxes = decode(pred)
            bboxes = postprocess(bboxes, params)
            result, pred_index, gt_index = calculate_metrics(bboxes, gt_bboxes)
            # print(result, len(bboxes))
            # print(result, len(gt_bboxes))
            precision.update(result, len(bboxes))
            recall.update(result, len(gt_bboxes))
            visualize(im, bboxes, imname)
        except:
            pass
    print(precision.compute())
    print(recall.compute())


"""
orig
    precision: 0.5128
    recall: 0.3194
tempered
    precision: 0.3428
    recall: 0.3156
tempered-vggblocktemper1:
    precision: 0.5312
    recall: 0.3150
"""

