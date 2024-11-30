import pandas as pd
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from docx2pdf import convert
from pdf2image import convert_from_path
import torch.nn.functional as F
import math
import ast
import json
import cv2

def pad_channels(X, h = 224, w = 224, c = 0):
    if c==0: #the user has not specified a number of channels
        max_c = int(max(x.shape[0] for x in X))
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

def resize_transform(tensor):
    resize = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return resize(tensor)

def main():
    train = pd.read_csv("Preprocessed_data_CNN/Train_normalized_CV.csv")
    test = pd.read_csv("Preprocessed_data_CNN/Test_normalized_CV.csv")
    
    train = train.drop("Unnamed: 0", axis = 1)
    test = test.drop("Unnamed: 0", axis = 1)
    
    y_train = train.drop("File name", axis = 1)
    y_test = test.drop("File name", axis = 1)
    
    
    
    path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets" #Add the path to your dataset folder here
    
    resize = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    X_train = []
    for i in range(0, len(train["File name"])):
        #print(i)
        if "docx" in train["File name"][i]:
            train.loc[i, "File name"] = train["File name"][i][:-5] + ".png"
            # continue
        if "gif" in train["File name"][i]:
            train.loc[i, "File name"] = train["File name"][i][:-4] + ".png"
            # continue
        if 'pdf' in train["File name"][i]:
            train.loc[i, "File name"] = train["File name"][i][:-4] + ".png"
        # print(i)
        image_path = path + "\\" + train["File name"][i]
        image = Image.open(image_path)
        tensor = resize(image)
        X_train.append(tensor)
        
    X_test = []
    for i in range(0, len(test["File name"])):
        #print(i)
        if "docx" in test["File name"][i]:
            test.loc[i, "File name"] = test["File name"][i][:-5] + ".png"
            # continue
        if "gif" in test["File name"][i]:
            test.loc[i, "File name"] = test["File name"][i][:-4] + ".png"
            # continue
        if 'pdf' in test["File name"][i]:
            test.loc[i, "File name"] = test["File name"][i][:-4] + ".png"
        # print(i)
        image_path = path + "\\" + test["File name"][i]
        image = Image.open(image_path)
        tensor = resize(image)
        X_test.append(tensor)
    
    
    # ### Now we need to find the max. value of channels and pad the data accordingly.
    
    X_train_padded = pad_channels(X_train)
    #print((X_train_padded[0].shape[0]))
    X_test_padded = pad_channels(X_test, c = int(X_train_padded[0].shape[0]))
    
    X_train_tensor = torch.stack(X_train_padded)
    X_test_tensor = torch.stack(X_test_padded)
    
    
    torch.save(X_train_tensor, "X_train_tensor.pt")
    torch.save(X_test_tensor, "X_test_tensor.pt")
    
    
    X_test_tensor.shape

if __name__ == "__main__":
    main()