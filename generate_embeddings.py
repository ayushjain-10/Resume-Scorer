import torch
import torch.nn as nn
import pad_data as pad
import os
from PIL import Image

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

def hook_fn(module, input, output):
    intermediate_outputs[module] = output

def main():
	state_dict = torch.load("ResumeRater/shallow_cnn.pth")
	state_dict.keys()
	model = shallow_CNN()
	model.load_state_dict(state_dict)

	intermediate_outputs = {}



	# Register hooks for specific layers
	for name, layer in model.named_modules():
	    if name in ["fc2"]:  # Specify layer names to hook
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
	            tensor = pad.resize_transform(image)
	            # print(tensor.shape)
	            tensors[image_path] = (tensor)
	            # tensor = pad.pad_channels(tensor)
	            # with torch.no_grad():
	            #     output = model(input_tensor)	


	combined_tensor = (list(tensors.values()))

	c = max([i.shape for i in combined_tensor])[0]

	padded_combined = pad.pad_channels(combined_tensor, c = c)
	padded_tensor = torch.stack(padded_combined)


	with torch.no_grad():
	    output = model(padded_tensor)

	embeddings = intermediate_outputs[list(intermediate_outputs.keys())[0]]