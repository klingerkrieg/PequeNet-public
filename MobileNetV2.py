#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# DeepLabV3+MobilenetV2
from torchvision.models import mobilenet_v2
# Adaptar para usar como DeepLab
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import torch.nn as nn

from Trainer import Trainer
from SemanticPequeNet import SemanticTrainer

#jupyter nbconvert --to script MobileNetV2.ipynb


# In[ ]:


class MobileNetBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        mobilenet = mobilenet_v2(weights='DEFAULT')


        if in_channels != 3:
            # Modificar a primeira camada para aceitar 1 canal
            first_conv = mobilenet.features[0][0]  # Conv2d
            mobilenet.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )

        self.features = mobilenet.features
        self.out_channels = 1280

    def forward(self, x):
        x = self.features(x)
        return {"out": x} 


class MobileNetV2Trainer(Trainer):

    def get_model_output(self,images):
        return self.model(images)['out']

class MobileNetV2SemanticTrainer(SemanticTrainer):

    def get_model_output(self,images):
        return self.model(images)['out']


def getDeepLabV3_MobileNetV2(num_classes, in_channels=3):
    backbone = MobileNetBackbone(in_channels=in_channels)
    return DeepLabV3(backbone=backbone, classifier=DeepLabHead(1280, num_classes))

