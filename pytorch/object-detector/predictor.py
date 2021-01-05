# -*- coding: utf-8 -*-

# -- stdlib --
from io import BytesIO
import time

# -- third party --
from PIL import Image
from torchvision import models, transforms
import requests
import torch

import openbayes_serving as serv

# -- own --


# -- code --
class PythonPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")

        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
        model = model.to(self.device)
        model.eval()

        self.preprocess = transforms.Compose([transforms.ToTensor()])

        with open("coco_labels.txt") as f:
            self.coco_labels = f.read().splitlines()

        self.model = model

    def predict(self, json):
        threshold = float(json["threshold"])

        b4 = time.time()
        image = requests.get(json["url"]).content
        af = time.time()
        serv.emit_event("image-download-time.txt", str(af - b4))
        serv.emit_event("image-raw.jpg", image)  # 假设是 jpg

        img_pil = Image.open(BytesIO(image))
        img_preprocessed = self.preprocess(img_pil)
        img_tensor = img_preprocessed.to(self.device)
        img_tensor.unsqueeze_(0)

        with torch.no_grad():
            pred = self.model(img_tensor)

        predicted_class = [self.coco_labels[i] for i in pred[0]["labels"].cpu().tolist()]
        predicted_boxes = [
            [(i[0], i[1]), (i[2], i[3])] for i in pred[0]["boxes"].detach().cpu().tolist()
        ]
        predicted_score = pred[0]["scores"].detach().cpu().tolist()
        predicted_t = [predicted_score.index(x) for x in predicted_score if x > threshold]
        if len(predicted_t) == 0:
            serv.emit_event("result-is-empty", "result-is-empty")
            return [], []

        predicted_t = predicted_t[-1]
        predicted_boxes = predicted_boxes[: predicted_t + 1]
        predicted_class = predicted_class[: predicted_t + 1]

        serv.emit_event("result.json", {
            'boxes': predicted_boxes,
            'classes': predicted_class,
        })

        return predicted_boxes, predicted_class


if __name__ == '__main__':
    serv.run(PythonPredictor)
