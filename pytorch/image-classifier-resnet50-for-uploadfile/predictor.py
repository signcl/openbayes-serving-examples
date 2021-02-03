import torch
import cv2
import numpy as np
import json
from torchvision import models, transforms
import openbayes_serving as serv


def get_file_image(raw):
    image = np.asarray(bytearray(raw), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class PythonPredictor:
    def __init__(self, config):
        # 加载分类元数据
        classes = json.load(open('classes.json'))
        self.idx2label = [classes[str(k)][1] for k in range(len(classes))]

        # 指定模型文件名称
        model_name = 'resnet50.pt'

        # 加载模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet50()
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()
        self.model = self.model.to(self.device)

        # 图像预处理，包括 normalization 和 resize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def predict(self, payload):
        # 获取图片内容
        image = get_file_image(payload)

        # 事件记录演示
        serv.emit_event(f"image.jpg", cv2.imencode('.jpg', image[:, :, ::-1])[1].tobytes())

        # 图片预处理
        image = self.transform(image)
        image = torch.tensor(image.numpy()[np.newaxis, ...])

        # 推理
        results = self.model(image.to(self.device))

        # 获取结果前五
        top5_idx = results[0].sort()[1][-5:]

        # 获取前五分类的名称
        top5_labels = [self.idx2label[idx] for idx in top5_idx]
        top5_labels = top5_labels[::-1]

        return top5_labels


if __name__ == '__main__':
    serv.run(PythonPredictor)