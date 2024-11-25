import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
from docx2pdf import convert
from pdf2image import convert_from_path
import cv2

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

train = train.drop(["Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned", "Awards/honours", "Avg quality score"], axis = 1)
test = test.drop(["Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned", "Awards/honours", "Avg quality score"], axis = 1)

for i in train.columns:
    if i== "File name":
        continue
    for j in range(0, len(train[i])):
        train.loc[j, i] = float(train[i][j])

for i in train.columns:
    # train[i] = float(train[i])
    if i == "File name":
        continue
    train[i] = (train[i] - train[i].min()) / (train[i].max() - train[i].min())

for i in test.columns:
    if i== "File name":
        continue
    for j in range(0, len(test[i])):
        test.loc[j, i] = float(test[i][j])

for i in test.columns:
    if i == "File name":
        continue
    test[i] = (test[i] - test[i].min()) / (test[i].max() - test[i].min())


transform = transforms.Compose([
    transforms.ToTensor() 
])

files = os.listdir(path + "Scrapped_Resumes\\")

for i in files:
    path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets\\Scrapped_Resumes\\" + i #replace with the path to data here
    subdir = os.listdir(path)
    for j in subdir:
        if "docx" in j:
            # print(j)
            convert(path + "\\" + j)
            # break

for i in files:
    path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets\\Scrapped_Resumes\\" + i #replace with the path to data here
    subdir = os.listdir(path)
    print(i)
    for j in subdir:
        if "pdf" in j:
            print(j)
            # print(j)
            pdf_path = (path + "\\" + j)
            # break
            images = convert_from_path(pdf_path, dpi=300) 
            for k, image in enumerate(images):
                image.save(pdf_path[:-4]+f"_{k + 1}.png", "PNG")


for i in files:
    path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets\\Scrapped_Resumes\\" + i #replace with path to data here
    subdir = os.listdir(path)
    print(i)
    for j in subdir:
        if "pdf" in j:
            print(j)
            # print(j)
            pdf_path = (path + "\\" + j)
            without_ext = j[:-4]
            k = 1
            imgs = []
            while (without_ext + f"_{k}.png") in subdir:
                imgs.append(cv2.imread(path + "\\" + without_ext + f"_{k}.png"))
                k += 1
            img_concat = cv2.vconcat(imgs)
            cv2.imwrite(path + "\\" + without_ext + ".png", img_concat)

for i in files:
    path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets\\Scrapped_Resumes\\" + i
    subdir = os.listdir(path)
    print(i)
    for j in subdir:
        if "docx" in j:
            os.remove(path + "\\" + j)
        if "pdf" in j:
            print(j)
            # print(j)
            pdf_path = (path + "\\" + j)
            os.remove(pdf_path)
            without_ext = j[:-4]
            k = 2
            imgs = []
            while (without_ext + f"_{k}.png") in subdir:
                os.remove(path + "\\" + without_ext + f"_{k}.png")
                k += 1


train = train.drop("Unnamed: 0", axis = 1)
test = test.drop("Unnamed: 0", axis = 1)

for i in range(0, len(train["File name"])):
    if "docx" in train["File name"][i]:
        train.loc[i, "File name"] = train["File name"][i][:-5] + ".png"
    if "gif" in train["File name"][i]:
        train.loc[i, "File name"] = train["File name"][i][:-4] + ".png"

train.to_csv("Preprocessed_data_CNN/Train_normalized_CV.csv")

for i in range(0, len(test["File name"])):
    if "pdf" in test["File name"][i]:
    if "docx" in test["File name"][i]:
        test.loc[i, "File name"] = test["File name"][i][:-5] + ".png"
        docx += 1
    if "gif" in test["File name"][i]:
        test.loc[i, "File name"] = test["File name"][i][:-4] + ".png"
        gif += 1

test.to_csv("Preprocessed_data_CNN/Test_normalized_CV.csv")

