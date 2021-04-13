import torch
from torchvision import transforms as T
from torchvision.transforms.transforms import ToPILImage
from transforms import RandomHorizontalFlip, Sequence


class Config:
    # preprocess
    insize = [416, 416]
    channels = 3
    downscale = 4
    sigma = 2.65
    # TODO Horizontal flip
    # TODO Remove landmark
    # TODO Use more transform
    train_cotransforms = Sequence([RandomHorizontalFlip()])
    train_transforms = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    ])

    # dataset
    # dataroot = '/content/WIDER_train/images'
    # annfile = '/content/retinaface_gt_v1.1/train/label.txt'
    dataroot = 'data/WIDER_train/images'
    annfile = 'data/retinaface_gt_v1.1/train/label.txt'

    valdataroot = 'data/WIDER_val/images'
    valannfile = 'data/retinaface_gt_v1.1/val/label.txt'

    # checkpoints
    checkpoints = 'checkpoints'
    restore = False
    restore_model = 'final.pth'

    # training
    epoch = 90
    lr = 5e-4
    batch_size = 32
    pin_memory = True
    num_workers = 4
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # inference
    threshold = 0.4
