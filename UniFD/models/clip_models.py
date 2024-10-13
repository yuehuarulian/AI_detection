from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14@336px" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")# self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc1 = nn.Linear(CHANNELS[name], 2048)  
        self.fc2 = nn.Linear(2048, num_classes)  # 输出层

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        x = self.fc1(features)
        x = torch.relu(x)  # 激活函数
        return self.fc2(x)



