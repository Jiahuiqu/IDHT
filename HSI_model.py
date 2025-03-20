import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import numpy as np
from scipy.io import loadmat

class IDHT(nn.Module):
    def __init__(self, num_class=2):
        super(IDHT, self).__init__()

        self.f = []

        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(224, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.fl = nn.Sequential(*self.f)
        # ISD
        self.ISD = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, num_class, bias=True))

    def forward(self, x):
        x = self.fl(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.ISD(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)




if __name__=="__main__":
    for name, module in resnet50().named_children():
        print(name, module)

