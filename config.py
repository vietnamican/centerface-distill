import torch
from torchvision import transforms as T


class Config:
    # preprocess
    insize = [416, 416]
    channels = 3
    downscale = 4
    sigma = 2.65

    train_transforms = T.Compose([
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # dataset
    # dataroot = '/content/WIDER_train/images'
    # annfile = '/content/retinaface_gt_v1.1/train/label.txt'
    dataroot = 'data/WIDER_train/images'
    annfile = 'data/retinaface_gt_v1.1/train/label.txt'

    valdataroot = 'data/WIDER_val/images'
    valannfile = 'data/retinaface_gt_v1.1/val/label.txt'
    # valdataroot = '/content/WIDER_val/images'
    # valannfile = '/content/retinaface_gt_v1.1/val/label.txt'

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inference
    threshold = 0.4
