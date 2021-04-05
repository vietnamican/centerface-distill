from time import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F
from PIL import Image

from models.model import Model
from models.mobilenetv2 import MobileNetV2VGGBlockTemper1, MobileNetV2

from api import postprocess, preprocess, calculate_metrics, detect, decode

torch.set_grad_enabled(False)

def load_model():
    device = 'cpu'
    checkpoint_path = 'checkpoints/checkpoint-epoch=93-val_loss=0.0497.ckpt'
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
    net.eval()

    return net

def draw_box(net, frame):
    # print(frame.shape)
    image = Image.fromarray(frame)
    new_im, params = preprocess(image)
    pred = detect(net, new_im)
    bboxes, landmarks = decode(pred)
    new_im = np.array(new_im)
    if bboxes.shape[0] > 0:
        # print("has {} face".format(bboxes.shape[0]))
        bboxes, landmarks = postprocess(bboxes, landmarks, params)
        for bbox in bboxes:
            left, top, right, bottom = map(int, bbox)
            cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
    return frame

if __name__ == "__main__":
    # net = load_model()
    # define a video capture object
    vid = cv2.VideoCapture(0)
    net = load_model()
    while(True):
        
        # Capture the video frame
        # by frame
        start_frame = time()
        ret, frame = vid.read()
        end_frame = time()
        start_draw = time()
        frame = draw_box(net, frame)
        end_draw = time()
        start_show = time()
        frame = np.array(frame, dtype=np.uint8)
    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_show = time()
        t_capture = end_frame-start_frame
        t_draw = end_draw-start_draw
        t_show = end_show-start_show
        print('t.capture: {:.4f}, t.draw: {:.4f}, t.show: {:.4f}'.format(t_capture, t_draw, t_show))
        print('time {:.4f} vs {:.4f}'.format(t_capture + t_draw + t_show, end_show - start_frame))
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

