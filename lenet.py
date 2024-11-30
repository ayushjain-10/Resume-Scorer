import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

X_train = torch.load("X_train_tensor.pt")
X_test = torch.load("X_test_tensor.pt")


train_df = pd.read_csv("C:\\Users\\brooklyn\\OneDrive\\Documents\\School\\AI\\ResumeRater\\Preprocessed_data_CNN\\Train_normalized_CV.csv")
train_df = train_df.drop("Unnamed: 0", axis = 1)

test_df = pd.read_csv("C:\\Users\\brooklyn\\OneDrive\\Documents\\School\\AI\\ResumeRater\\Preprocessed_data_CNN\\Test_normalized_CV.csv")
test_df = test_df.drop("Unnamed: 0", axis = 1)

train_labels = train_df.drop(["File name", "Avg Format score"], axis = 1)
y_train = train_labels.values
y_train = torch.tensor(y_train)

test_labels = test_df.drop(["File name", "Avg Format score"], axis = 1)
y_test = test_labels.values
y_test = torch.tensor(y_test)

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
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x

lenet = LeNet()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lenet.parameters())

start_time = datetime.now()
num_epochs = 50
training_loss = []

for epoch in range(num_epochs):
    running_loss = 0.0
    # for i in range(0, y_train.shape[0]):
    inputs = X_train.float()
    labels = y_train.float()

    #print(inputs.shape)
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = lenet(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(f"Epoch {epoch+1}:")
    end_time = datetime.now()
    print(f"Time taken to get here: {end_time-start_time}")
    print(f"Training loss: {running_loss}")
    training_loss.append(running_loss)

print('Finished Training')

torch.save(lenet.state_dict(), 'lenet_5.pth')

mse_loss = nn.MSELoss()

with torch.no_grad():
	preds = lenet(X_test)
loss = mse_loss(preds, y_test)


print("MSE Loss: ", loss.item())
print(preds)
