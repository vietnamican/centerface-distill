import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from utils import VisionKit


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    radius = round(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw(hm, name):
    hm[hm > 0.1] = 1
    np.savetxt(name, hm, fmt='%d')


class WiderFace(Dataset, VisionKit):

    def __init__(self, dataroot, annfile, sigma, downscale, insize, transforms=None, mode='train'):
        """
        Args:
            dataroot: image file directory
            annfile: the retinaface annotations txt file
            sigma: control the spread of center point
            downscale: aka down-sample factor. `R` in paper CenterNet 
            insize: input size
            transforms: torchvision.transforms.Compose object, refer to `config.py`
        """
        self.root = dataroot
        self.sigma = sigma
        self.downscale = downscale
        self.insize = insize
        self.transforms = transforms
        self.namelist, self.annslist = self.parse_annfile(annfile)
        self.split_index = -1000
        if mode == 'train':
            self.namelist, self.annslist = \
                self.namelist[:self.split_index], self.annslist[:self.split_index]
        else:
            self.namelist, self.annslist = self.namelist[self.split_index:], self.annslist[self.split_index:]

    def __getitem__(self, idx):
        path = osp.join(self.root, self.namelist[idx])
        im = cv2.imread(path)
        im = im[:,:,::-1]
        anns = self.annslist[idx]
        im, bboxes, landmarks = self.preprocess(im, anns)
        hm = self.make_heatmaps(im, bboxes, landmarks, self.downscale)
        if self.transforms is not None:
            im = self.transforms(im)
        return im, hm

    def __len__(self):
        return len(self.annslist)

    def xywh2xyxy(self, bboxes):
        _bboxes = bboxes.copy()
        _bboxes[:, 2] += _bboxes[:, 0]
        _bboxes[:, 3] += _bboxes[:, 1]
        return _bboxes

    def preprocess(self, im, anns):
        bboxes = anns[:, :4]
        bboxes = self.xywh2xyxy(bboxes)
        landmarks = anns[:, 4:-1]
        im, bboxes, landmarks, * \
            _ = self.letterbox(im, self.insize, bboxes, landmarks)
        return im, bboxes, landmarks

    def make_heatmaps(self, im, bboxes, landmarks, downscale):
        """make heatmaps for one image
        Returns: 
            Heatmap in numpy format with some channels
            #0 for heatmap      
            #1 for offset x     #2 for offset y
            #3 for width        #4 for height
            #5-14 for five landmarks
        """
        height, width = im.shape[:2]
        width = int(width / downscale)
        height = int(height / downscale)
        res = np.zeros([15, height, width], dtype=np.float32)

        grid_x = np.tile(np.arange(width), reps=(height, 1))
        grid_y = np.tile(np.arange(height), reps=(width, 1)).transpose()

        for bbox, landmark in zip(bboxes, landmarks):
            try:
                # 0 heatmap
                left, top, right, bottom = map(
                    lambda x: int(x / downscale), bbox)
                x = (left + right) // 2
                y = (top + bottom) // 2
                grid_dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
                heatmap = np.exp(-0.5 * grid_dist / self.sigma ** 2)
                res[0] = np.maximum(heatmap, res[0])
                # 1, 2 center offset
                original_x = (bbox[0] + bbox[2]) / 2
                original_y = (bbox[1] + bbox[3]) / 2

                res[1][y, x] = original_x / downscale - x
                res[2][y, x] = original_y / downscale - y
                # 3, 4 size
                width = right - left
                height = bottom - top
                res[3][y, x] = np.log(width + 1e-4)
                res[4][y, x] = np.log(height + 1e-4)
                # 5-14 landmarks
                if landmark[0] == -1:
                    continue
                original_width = bbox[2] - bbox[0]
                original_height = bbox[3] - bbox[1]
                skip = 3
                lm_xs = landmark[0::skip]
                lm_ys = landmark[1::skip]
                lm_xs = (lm_xs - bbox[0]) / original_width
                lm_ys = (lm_ys - bbox[1]) / original_height
                for i, lm_x, lm_y in zip(range(5, 14, 2), lm_xs, lm_ys):
                    res[i][y, x] = lm_x
                    res[i+1][y, x] = lm_y
            except:
                pass
        return res

    def parse_annfile(self, annfile):
        lines = open(annfile, 'r', encoding='utf-8').read()
        data = lines.split('#')[1:]
        data = map(lambda record: record.split('\n'), data)
        namelist = []
        annslist = []
        for record in data:
            record = [r.strip() for r in record if r]
            name, anns = record[0], record[1:]
            nrow = len(anns)
            anns = np.loadtxt(anns).reshape(nrow, -1)
            namelist.append(name)
            annslist.append(anns)
        return namelist, annslist


class WiderFaceVal(Dataset, VisionKit):

    def __init__(self, dataroot, annfile, sigma, downscale, insize, transforms=None, mode='train'):
        """
        Args:
            dataroot: image file directory
            annfile: the retinaface annotations txt file
            sigma: control the spread of center point
            downscale: aka down-sample factor. `R` in paper CenterNet 
            insize: input size
            transforms: torchvision.transforms.Compose object, refer to `config.py`
        """
        self.root = dataroot
        self.sigma = sigma
        self.downscale = downscale
        self.insize = insize
        self.transforms = transforms
        self.namelist, self.annslist = self.parse_annfile(annfile)
        # self.split_index = -1000
        # if mode == 'train':
        #     self.namelist, self.annslist = \
        #         self.namelist[:self.split_index], self.annslist[:self.split_index]
        # else:
        #     self.namelist, self.annslist = self.namelist[self.split_index:], self.annslist[self.split_index:]

    def __getitem__(self, idx):
        path = osp.join(self.root, self.namelist[idx])
        im = cv2.imread(path)
        im = im[:,:,::-1]
        anns = self.annslist[idx]
        im, bboxes, landmarks = self.preprocess(im, anns)
        hm = self.make_heatmaps(im, bboxes, landmarks, self.downscale)
        if self.transforms is not None:
            im = self.transforms(im)
        return im, hm

    def __len__(self):
        return len(self.annslist)

    def xywh2xyxy(self, bboxes):
        _bboxes = bboxes.copy()
        _bboxes[:, 2] += _bboxes[:, 0]
        _bboxes[:, 3] += _bboxes[:, 1]
        return _bboxes

    def preprocess(self, im, anns):
        bboxes = anns[:, :4]
        bboxes = self.xywh2xyxy(bboxes)
        im, bboxes, * \
            _ = self.letterbox(im, self.insize, bboxes)
        return im, bboxes

    def make_heatmaps(self, im, bboxes, downscale):
        """make heatmaps for one image
        Returns: 
            Heatmap in numpy format with some channels
            #0 for heatmap      
            #1 for offset x     #2 for offset y
            #3 for width        #4 for height
        """
        width, height = im.size
        width = int(width / downscale)
        height = int(height / downscale)
        res = np.zeros([5, height, width], dtype=np.float32)

        grid_x = np.tile(np.arange(width), reps=(height, 1))
        grid_y = np.tile(np.arange(height), reps=(width, 1)).transpose()

        for bbox in bboxes:
            try:
                # 0 heatmap
                left, top, right, bottom = map(
                    lambda x: int(x / downscale), bbox)
                x = (left + right) // 2
                y = (top + bottom) // 2
                grid_dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
                heatmap = np.exp(-0.5 * grid_dist / self.sigma ** 2)
                res[0] = np.maximum(heatmap, res[0])
                # 1, 2 center offset
                original_x = (bbox[0] + bbox[2]) / 2
                original_y = (bbox[1] + bbox[3]) / 2

                res[1][y, x] = original_x / downscale - x
                res[2][y, x] = original_y / downscale - y
                # 3, 4 size
                width = right - left
                height = bottom - top
                res[3][y, x] = np.log(width + 1e-4)
                res[4][y, x] = np.log(height + 1e-4)
            except:
                pass
        return res

    def parse_annfile(self, annfile):
        lines = open(annfile, 'r', encoding='utf-8').read()
        data = lines.split('#')[1:]
        data = map(lambda record: record.split('\n'), data)
        namelist = []
        annslist = []
        for record in data:
            record = [r.strip() for r in record if r]
            name, anns = record[0], record[1:]
            nrow = len(anns)
            anns = np.loadtxt(anns).reshape(nrow, -1)
            namelist.append(name)
            annslist.append(anns)
        return namelist, annslist


if __name__ == "__main__":
    from config import Config as cfg
    import matplotlib.pyplot as plt
    dataroot = '/data/WIDER_train/images'
    annfile = '/data/retinaface_gt_v1.1/train/label.txt'
    dataset = WiderFace(cfg.dataroot, cfg.annfile, cfg.sigma,
                        cfg.downscale, cfg.insize, cfg.train_transforms)
    ids = 10969
    print(dataset.namelist[ids])
