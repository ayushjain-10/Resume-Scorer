import torch
import pandas as pd
import numpy as np
import warnings
import torch.nn as nn
from datetime import datetime

X_train = torch.load("X_train_tensor.pt")
X_test = torch.load("X_test_tensor.pt")

train_df = pd.read_csv("ResumeRater\\Preprocessed_data_CNN\\Train_normalized_CV.csv")
train_df = train_df.drop("Unnamed: 0", axis = 1)

test_df = pd.read_csv("ResumeRater\\Preprocessed_data_CNN\\Test_normalized_CV.csv")
test_df = test_df.drop("Unnamed: 0", axis = 1)

train_labels = train_df.drop(["File name", "Avg Format score"], axis = 1)
y_train = train_labels.values
y_train = torch.tensor(y_train)

test_labels = test_df.drop(["File name", "Avg Format score"], axis = 1)
y_test = test_labels.values
y_test = torch.tensor(y_test)

class VGG_16(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
        	nn.Dropout(0.5),
        	nn.Linear(7*7*512, 4096),
        	nn.ReLU())

        self.fc2 = nn.Sequential(
        	nn.Dropout(0.5),
        	nn.Linear(4096, 4096),
        	nn.ReLU())  

        self.fc3= nn.Sequential(
        	nn.Linear(4096, 3))


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim = 1)
        return x


vgg = VGG_16()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg.parameters())

start_time = datetime.now()
num_epochs = 1
training_loss = []

for epoch in range(num_epochs):
    running_loss = 0.0
    # for i in range(0, y_train.shape[0]):
    inputs = X_train
    labels = y_train

    print(inputs.shape)
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = vgg(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(f"Epoch {epoch+1}:")
    end_time = datetime.now()
    print(f"Time taken to get here: {end_time-start_time}")
    print(f"Training loss: {running_loss}")
    training_loss.append(running_loss)
    # if epoch%50 == 0:
    #     torch.save(vgg.state_dict(),'vgg16.pth')
        # with open("vgg16.txt","w") as f:
        #     f.write(str(training_loss))

print('Finished Training')

torch.save(vgg.state_dict(), 'vgg_16.pth')

mse_loss = nn.MSELoss()

with torch.no_grad():
	preds = vgg(X_test)
loss = mse_loss(preds, y_test)

print("MSE Loss: ", loss.item())
