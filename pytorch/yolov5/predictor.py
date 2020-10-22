# -*- coding: utf-8 -*-

# -- stdlib --
# -- third party --
import cv2
import numpy as np
import requests
import torch

import openbayes_serving as serv

# -- own --
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


# -- code --
def get_url_image(url_image):
    """
    Get numpy image from URL image.
    """
    resp = requests.get(url_image, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


class Predictor:
    def __init__(self):
        self.weights = 'yolov5s.pt'
        imgsz = 640

        # Initialize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess(self, img0):
        img = letterbox(img0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img = img / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def postprocess(self, output, threshold, img0, img):
        pred = non_max_suppression(output, threshold, threshold, classes=[], agnostic=False)
        boxes, classes = [], []
        # Process detections
        for det in pred:  # detections per image
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, class_type in reversed(det):
                    classes.append(self.names[int(class_type)])
                    boxes.append(((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))))

        return boxes, classes

    def predict(self, json):
        threshold = float(json["threshold"])
        img_url = json["url"]

        # process the input
        img0 = get_url_image(img_url)
        img = self.preprocess(img0)

        # run predictions
        output = self.model(img, augment=False)[0]

        # postprocess
        predicted_boxes, predicted_classes = self.postprocess(output, threshold, img0, img)

        return predicted_boxes, predicted_classes


if __name__ == '__main__':
    serv.run(Predictor)
