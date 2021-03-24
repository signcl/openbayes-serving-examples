# -*- coding: utf-8 -*-

# -- stdlib --
import logging

# -- third party --
from numpy import random
from torch import nn
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# -- own --
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import openbayes_serving as serv


# -- code --
log = logging.getLogger('predictor')


class Predictor:
    def __init__(self):
        self.cap = None
        self.current_frame = None
        self.config = {
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'agnostic_nms': False,
        }

        weights = 'yolov5s.pt'
        self.imgsz = 640

        self.device = torch.device('cuda')

        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        assert self.imgsz % self.stride == 0
        model.half()  # to FP16

        # Set Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once

        self.model = model

    def predict(self, json):
        if self.cap is None:
            return {}

        with_orig = json.get('withOriginal')
        with_tagged = json.get('withTagged')

        rst, orig, tagged = self.detect(with_tagged_image=with_tagged)
        resp = {'detections': rst}

        if with_orig:
            _, buf = cv2.imencode('foo.jpg', orig)
            resp['original'] = buf.tobytes()

        if with_tagged:
            _, buf = cv2.imencode('foo.jpg', tagged)
            resp['tagged'] = buf.tobytes()

        return resp

    def run(self):
        cap = cv2.VideoCapture("rtsp://localhost:8554/basketball")
        if not cap:
            raise Exception('打开摄像头失败')

        self.cap = cap
        log.info('成功打开摄像头')

        while cap.isOpened():
            if not cap.grab():
                raise Exception('抓取摄像头失败')

        cap.release()

    def capture(self):
        ret, img = self.cap.retrieve()
        if not ret:
            return self.current_frame

        self.current_frame = img
        return img

    def detect(self, with_tagged_image=False):
        img0 = self.capture()
        imgsz = self.imgsz

        img = letterbox(img0, imgsz, auto=True, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        # img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.config['conf_thres'],
                                   self.config['iou_thres'],
                                   agnostic=self.config['agnostic_nms'])
        det = pred[0]

        # Process detections
        rst = []
        tagged = None
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            rst = [
                {'coords': [
                    [int(pos[0]), int(pos[1])],
                    [int(pos[2]), int(pos[3])],
                  ],
                 'confidence': float(conf),
                 'class': self.names[int(cls)]}
                for t in reversed(det)
                for *pos, conf, cls in [map(float, t)]
            ]

            # Write results
            if with_tagged_image:
                tagged = img0.copy()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, tagged, label=label, color=self.colors[int(cls)], line_thickness=3)

        return rst, img0, tagged


if __name__ == "__main__":
    serv.run(Predictor)
