import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F
from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlockTemper1

from api import postprocess, preprocess, calculate_metrics, detect, decode
# parser = argparse.ArgumentParser(description='human matting')
# parser.add_argument('--model', default='./', help='preTrained model')
# parser.add_argument('--size', type=int, default=256, help='input size')
# parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

torch.set_grad_enabled(False)
def load_model():
    model = Model(MobileNetV2VGGBlockTemper1)
    device = 'cpu'
    checkpoint_path = 'checkpoint-epoch=139-val_loss=0.0515.ckpt'
    # checkpoint_path = 'checkpoints/final.pth'
    if device == 'cpu':
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    model.eval()
    model.migrate(state_dict, force=True, verbose=1)
    model.eval()
    
    return model

def seg_process(image, net):
    size=416
    # opencv
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC)
    # image_resize = (image_resize - (104., 112., 121.,)) / 255.0
    new_im, params = preprocess(image_resize)
    pred = detect(new_im)
    bboxes, landmarks = decode(pred)
    bboxes, landmarks = postprocess(bboxes, landmarks, params)
    
    # result, pred_index, gt_index = calculate_metrics(bboxes, gt_bboxes)

    t0 = time.time()

    return new_im


def camera_seg(net):

    videoCapture = cv2.VideoCapture(0)

    while(1):
        # get a frame
        ret, frame = videoCapture.read()
        frame = cv2.flip(frame,1)
        frame_seg = seg_process(frame, net)


        # show a frame
        cv2.imshow("capture", frame_seg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()

def main():

    model = load_model()
    camera_seg(model)


if __name__ == "__main__":
    main()
