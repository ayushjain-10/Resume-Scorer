import torch
import torch.nn as nn
#import ResumeRater.pad_data as pad
import pad_data
import generate_embeddings
import os
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
#from ResumeRater import generate_embeddings

class shallow_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
        	nn.Dropout(0.6),
        	nn.Linear(13*13*128, 1024),
        	nn.ReLU())

        self.fc2 = nn.Sequential(
        	nn.Dropout(0.6),
        	nn.Linear(1024, 512),
        	nn.ReLU())  

        self.fc3= nn.Sequential(
        	nn.Linear(512, 3))


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

state_dict = torch.load("shallow_cnn.pth")
model = shallow_CNN()
model.load_state_dict(state_dict)

intermediate_outputs = {}

def hook_fn(module, input, output):
    intermediate_outputs[module] = output

for name, layer in model.named_modules():
    if name in ["fc2"]:  
        layer.register_forward_hook(hook_fn)

init_path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets"
folders = os.listdir("C:\\Users\\Tanvi\\Desktop\\Resumes Datasets")
tensors = {}
for f in folders:
    #print(f)
    files = os.listdir(init_path + "\\" + f)
    for i in files:
        path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets\\" + f + "\\" + i #replace with the path to data here
        subdir = os.listdir(path)
        for j in subdir:
            image_path = path + "\\" + j
            if image_path in tensors.keys():
                continue
            image = Image.open(image_path)
            tensor = pad_data.resize_transform(image)
            # print(tensor.shape)
            tensors[image_path] = (tensor)
            # tensor = pad.pad_channels(tensor)
            # with torch.no_grad():
            #     output = model(input_tensor)

combined_tensor = (list(tensors.values()))
c = max([i.shape for i in combined_tensor])[0]
padded_combined = pad_data.pad_channels(combined_tensor, c = c)
padded_tensor = torch.stack(padded_combined)

with torch.no_grad():
    output = model(padded_tensor)

embeddings = intermediate_outputs[list(intermediate_outputs.keys())[0]]

objects = {}
for i in range(0, 11874):
    objects[list(tensors.keys())[i]] = embeddings[i]

class data_points():
    def __init__(self, file_name, embedding):
        self.file_name = file_name
        self.embedding = embedding

data_pts = []
for i in objects.keys():
    temp = data_points(i, objects[i])
    data_pts.append(temp)

embeddings = [pt.embedding for pt in data_pts]
embeddings = torch.stack(embeddings)


kmeans = KMeans(n_clusters=24, random_state=42, max_iter = 300)
kmeans.fit(embeddings.numpy())

# Creating CSV with clusters
results = {"File Name": [pt.file_name for pt in data_pts], "Cluster": kmeans.labels_.tolist()}
results_df = pd.DataFrame(results)
results_df.to_csv("resume_clusters.csv", index=False)
 
print("Cluster results saved to resume_clusters.csv!")
 
# Visualization using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings.numpy())

np.save("reduced_embeddings.npy", reduced_embeddings)
 
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap="tab10", s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("Resumes Clustering Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

for i, pt in enumerate(data_pts):
    plt.annotate(f"Resume {i}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.75)
 
plt.savefig("resume_clusters_visualization.png")
plt.show()


# Perform PCA for visualization
 



