import cv2
import sys
import tempfile
import openbayes_serving as serv
sys.path.append("ailab/Real-CUGAN")
from torch import nn as nn
from upcunet_v3 import RealWaifuUpScaler


class PythonPredictor:
    def __init__(self):
        self.ModelName="up2x-latest-no-denoise.pth" 
        self.upscaler = RealWaifuUpScaler(2, self.ModelName, half=True, device="cuda:0")
            
    def predict(self, data):
        f = tempfile.NamedTemporaryFile()
        f.write(data)
        f.seek(0)
 
        img = cv2.imread(f.name)[:, :, [2, 1, 0]]
        result = self.upscaler(img,tile_mode=5,cache_mode=2,alpha=1)
    
        _, output = cv2.imencode('.png', result[:, :, ::-1])
    
        return output.tobytes()
 

if __name__ == '__main__':
    serv.run(PythonPredictor)
