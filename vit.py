import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
import numpy as np
import pandas as pd
import warnings
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

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
    
    return x
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()

    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

    # Creating positional encoding
    pe = torch.zeros(max_seq_length, d_model)

    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
    x = torch.cat((tokens_batch,x), dim=1)
    x = x + self.pe

    return x
  
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2,-1)

        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V
        return attention
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)

    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.W_o(out)
    return out
  
class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    # Sub-Layer 1 Normalization
    self.ln1 = nn.LayerNorm(d_model)

    # Multi-Head Attention
    self.mha = MultiHeadAttention(d_model, n_heads)

    # Sub-Layer 2 Normalization
    self.ln2 = nn.LayerNorm(d_model)

    # Multilayer Perception
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )

  def forward(self, x):
    # Residual Connection After Sub-Layer 1
    out = x + self.mha(self.ln1(x))

    # Residual Connection After Sub-Layer 2
    out = out + self.mlp(self.ln2(out))

    return out
  
class VisionTransformer(nn.Module):
  def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
    super().__init__()

    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    self.d_model = d_model 
    self.n_classes = n_classes 
    self.img_size = img_size 
    self.patch_size = patch_size 
    self.n_channels = n_channels 
    self.n_heads = n_heads 

    self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    self.max_seq_length = self.n_patches + 1

    self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
    self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads) for _ in range(n_layers)])

    self.classifier = nn.Sequential(
        nn.Linear(self.d_model, self.n_classes),
        nn.Softmax(dim=-1)
    )

  def forward(self, images):
    x = self.patch_embedding(images)

    x = self.positional_encoding(x)

    x = self.transformer_encoder(x)
    
    x = self.classifier(x[:,0])

    return x
  
d_model = 768  # Dimension of model
n_classes = 2
img_size = (224, 224)  # Image size (224x224)
patch_size = (16, 16)  # Patch size (16x16)
n_channels = 3  
n_heads = 12 
n_layers = 12 

vit = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers)
optimizer = Adam(vit.parameters(), lr=alpha)
criterion = nn.CrossEntropyLoss()

start_time = datetime.now()
num_epochs = 1
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
    outputs = vit(inputs)
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

torch.save(vit.state_dict(), 'vit.pth')

mse_loss = nn.MSELoss()

with torch.no_grad():
	preds = vit(X_test)
loss = mse_loss(preds, y_test)

print("MSE Loss: ", loss.item())