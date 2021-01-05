# -*- coding: utf-8 -*-

# -- stdlib --
import json
import time

# -- third party --
from scipy.special import softmax
import cv2
import numpy as np
import requests

import openbayes_serving as serv

# -- own --


# -- code --
def get_url_image(url_image):
    """
    Get numpy image from URL image.
    """
    resp = requests.get(url_image).content
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def image_resize(image, width, height, inter=cv2.INTER_AREA):
    """
    Resize a numpy image.
    """
    dim = (width, height)

    if width is None or height is None:
        return image

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def preprocess(img_data):
    """
    Normalize input for inference.
    """
    # move pixel color dimension to position 0
    img = np.moveaxis(img_data, 2, 0)

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img.shape).astype("float32")
    for i in range(img.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # extend to batch size of 1
    norm_img_data = norm_img_data[np.newaxis, ...]
    return norm_img_data


def postprocess(results):
    """
    Eliminates all dimensions of size 1, softmaxes the input and then returns the index of the element with the highest value.
    """
    squeezed = np.squeeze(results)
    maxed = softmax(squeezed)
    result = np.argmax(maxed)
    return result


class Predictor:
    def __init__(self, onnx):
        # onnx client
        self.onnx = onnx

        # for image classifiers
        classes = json.load(open('imagenet_class_index.json'))
        self.image_classes = [classes[str(k)][1] for k in range(len(classes))]
        self.resize_value = 224

    def predict(self, json):
        # get request params
        img_url = json["url"]

        # process the input
        b4 = time.time()
        img = get_url_image(img_url)
        af = time.time()

        serv.emit_event("image-download-time.txt", str(af - b4))
        serv.emit_event("image-raw.jpg", cv2.imencode('.jpg', img)[1].tobytes())

        img = image_resize(img, height=self.resize_value, width=self.resize_value)
        img = preprocess(img)

        # predict
        results = self.onnx.run(None, {'data': img})[0]

        # interpret result
        result = postprocess(results)
        predicted_label = self.image_classes[result]

        serv.emit_event("result.json", {"label": predicted_label})

        return {"label": predicted_label}


if __name__ == '__main__':
    serv.run(Predictor)
