import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils import VisionKit


class WiderFace(Dataset, VisionKit):
    EASY = [2]
    MEDIUM = [1]
    HARD = [0]
    ALL = [0, 1, 2]
    def __init__(self, dataroot, annfile, sigma, downscale, insize, transforms=None, proportion='easy'):
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
        if proportion == 'easy':
            self.proportion = self.EASY
        elif proportion == 'medium':
            self.proportion = self.MEDIUM
        elif proportion == 'hard':
            self.proportion = self.HARD
        else:
            self.proportion = self.ALL
        self.namelist, self.annslist = self.parse_annfile(annfile)

    def __getitem__(self, idx):
        path = osp.join(self.root, self.namelist[idx])
        im = Image.open(path)
        anns = self.annslist[idx]
        im, bboxes = self.preprocess(im, anns)
        hm = self.make_heatmaps(im, bboxes, self.downscale)
        if self.transforms is not None:
            im = self.transforms(im)
        return im, hm, self.namelist[idx], idx

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

    def _detemine_proporsion(self, cls):
        if int(cls) > 40:
            return 2
        elif int(cls) > 20:
            return 1
        else:
            return 0

    def parse_annfile(self, annfile):
        lines = open(annfile, 'r', encoding='utf-8').read()
        data = lines.split('#')[1:]
        data = map(lambda record: record.split('\n'), data)
        namelist = []
        annslist = []
        for record in data:
            record = [r.strip() for r in record if r]
            name, anns = record[0], record[1:]
            cls = name.split('--')[0]
            if self._detemine_proporsion(cls) in self.proportion:
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
