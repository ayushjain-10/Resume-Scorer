# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 22:58:35 2024

@author: brooklyn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
from PIL import Image
from datetime import datetime
from torchvision import transforms
import math


class LeNet(nn.Module):
    def __init__(self, input_channels=4, num_classes=3):
        super(LeNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # C1: Convolutional Layer 1 
            nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),  
            nn.ReLU(), 
            # S2: Average Pooling Layer
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: Convolutional Layer 2 
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16), 
            nn.ReLU(), 

            # S4: Average Pooling Layer
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  

            # FC5: Fully Connected Layer 1
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),  

            # FC6: Fully Connected Layer 2
            nn.Linear(120, 84),
            nn.ReLU(),  

            # Output layer
            nn.Linear(84, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x
    
def pad_channels(X, h = 224, w = 224, c = 0):
    if c==0: #the user has not specified a number of channels
        max_c = max(x.shape[0] for x in X)
    else:
        max_c = c
    max_vals = (w, w, h, h, max_c, max_c)
    X_padded = []
    for i in range(0, len(X)):
        # print(i)
        padded = []
        c, h, w = X[i].shape
        pad = (0, 0, 0, 0, math.floor((max_c-c)/2), math.ceil((max_c-c)/2))
        # print(pad)
        padded = F.pad(X[i], pad, "constant")
        X_padded.append(padded)
    return X_padded
    
resize = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

image = Image.open("C:\\Users\\brooklyn\\Downloads\\Merged_document.png")
tensor = resize(image)
X_raj = []
X_raj.append(tensor)

X_raj_padded = pad_channels(X_raj)
X_raj_tensor = torch.stack(X_raj_padded)

torch.save(X_raj_tensor, "X_raj_tensor.pt")

image2 = Image.open("C:\\Users\\brooklyn\\Downloads\\Ruhri Lee Resume - Nov 2024\\Ruhri Lee Resume - Nov 2024-1.png")
tensor2 = resize(image2)
X_ruhri = []
X_ruhri.append(tensor2)

X_ruhri_padded = pad_channels(X_ruhri)
X_ruhri_tensor = torch.stack(X_ruhri_padded)
X_ruhri_tensor = X_ruhri_tensor.repeat(1, 4, 1, 1)

torch.save(X_ruhri_tensor, "X_ruhri_tensor.pt")

lenet = LeNet()
lenet.load_state_dict(torch.load('lenet_5.pth', weights_only=True))
lenet.eval()

raj = torch.load('X_ruhri_tensor.pt', weights_only=True)

with torch.no_grad():
    preds = lenet(raj)
    
print(preds)