import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ResumeRater.pad_data import resize_transform, pad_channels

# Load pre-trained components
kmeans = joblib.load("kmeans_model.pkl")
embeddings = torch.load("embeddings.pth")
reduced_embeddings = np.load("reduced_embeddings.npy")
cluster_results = pd.read_csv("cluster_results.csv")

# Process uploaded CV
uploaded_cv_path = input("Enter the path to your CV image file: ")
uploaded_image = Image.open(uploaded_cv_path)
uploaded_tensor = resize_transform(uploaded_image)
uploaded_tensor = pad_channels([uploaded_tensor], c=embeddings.shape[1])
uploaded_tensor = torch.stack(uploaded_tensor)

# Load model for embedding extraction
state_dict = torch.load("ResumeRater/shallow_cnn.pth")
model = shallow_CNN()
model.load_state_dict(state_dict)
model.eval()

# extract embeddings
intermediate_outputs = {}
def hook_fn(module, input, output):
    intermediate_outputs[module] = output
model.fc2.register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(uploaded_tensor)

uploaded_embedding = intermediate_outputs[list(intermediate_outputs.keys())[0]][0]

# Predict the cluster
predicted_cluster = kmeans.predict(uploaded_embedding.unsqueeze(0).numpy())[0]
print(f"The uploaded CV belongs to cluster: {predicted_cluster}")

# Find similar resumes
similar_resumes = cluster_results[cluster_results["Cluster"] == predicted_cluster]["File Name"].values[:5]
print("\nTop 5 resumes from the same cluster:")
for resume in similar_resumes:
    print(resume)

# Visualization
plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap="tab10", s=50)
plt.scatter(uploaded_embedding[0], uploaded_embedding[1], c="red", s=100, label="Uploaded CV")
plt.colorbar(label="Cluster")
plt.legend()
plt.title("Resume Clusters (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
