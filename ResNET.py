import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchvision import models

# Load preprocessed training and testing data
X_train = torch.load(r'C:\Users\X1 YOGA\Downloads\X_train_tensor.pt')
X_test = torch.load(r'C:\Users\X1 YOGA\Downloads\X_test_tensor.pt')

# Convert 4-channel images (RGBA) to 3-channel (RGB)
X_train = X_train[:, :3, :, :]  # Keep only the first 3 channels (RGB)
X_test = X_test[:, :3, :, :]    # Same for the test data

train_df = pd.read_csv(r'C:\Users\X1 YOGA\Downloads\Train.csv')
test_df = pd.read_csv(r'C:\Users\X1 YOGA\Downloads\Test.csv')

columns_to_drop = ["File name", "Awards/honours", "Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned"]
train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
test_df = test_df.drop(columns=columns_to_drop, errors='ignore')

train_labels = train_df.drop("Avg Format score", axis=1)
y_train = torch.tensor(train_labels.values, dtype=torch.float32)

test_labels = test_df.drop("Avg Format score", axis=1)
y_test = torch.tensor(test_labels.values, dtype=torch.float32)

# Define ResNet Architecture
class ResNetCustom(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Load pretrained ResNet18 with updated 'weights' argument
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify the fully connected layer to fit the number of classes
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)  # Softmax for probabilities
        )

    def forward(self, x):
        return self.resnet(x)

# Initialize the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCustom(num_classes=y_train.shape[1]).to(device)
criterion = nn.MSELoss()  # Assuming regression task with scores
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed
training_loss = []
start_time = datetime.now()

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    inputs, labels = X_train.to(device).float(), y_train.to(device).float()

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    training_loss.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

end_time = datetime.now()
print(f"Training Completed. Time Taken: {end_time - start_time}")

# Save the trained model
torch.save(model.state_dict(), "resnet_custom.pth")

# Evaluation
model.eval()  
with torch.no_grad():
    inputs, labels = X_test.to(device).float(), y_test.to(device).float()
    predictions = model(inputs)
    loss = criterion(predictions, labels)

print(f"Test MSE Loss: {loss.item():.4f}")

# save test predictions
predictions = predictions.cpu().numpy()
test_df["Predictions"] = predictions.tolist()
test_df.to_csv("Test_with_Predictions.csv", index=False)
