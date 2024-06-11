import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Fashion MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Combine train and test datasets
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=64, shuffle=False)

#Building a CNN model to convert image to embeddings
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))      
        x = self.pool(x)               
        x = F.relu(self.conv2(x))      
        x = self.pool(x)               
        x = F.relu(self.conv3(x))      
        x = self.pool(x)               
        x = torch.flatten(x, 1)        
        x = F.relu(self.fc1(x))        
        x = self.fc2(x)                
        return x

model = CNN()

# Extract features
features = []
labels = []

model.eval()
with torch.no_grad():
    for images, lbls in combined_loader:
        output = model(images)
        features.append(output)
        labels.append(lbls)

features = torch.cat(features)
labels = torch.cat(labels)

# Convert features to numpy array for clustering
features_np = features.numpy()

# Cluster into 3 categories as we have 3 category in style category
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(features_np)
cluster_labels = kmeans.labels_

# Define category names
category_names = ['Formal Appearance', 'Casual Appearance', 'Intellectual Appearance']

# Map cluster labels to category names
category_mapping = {}
for i in range(3):
    category_mapping[i] = category_names[i]

# Function to get recommended items from a category
def recommend_items(category, num_items=5):
    category_idx = list(category_mapping.keys())[list(category_mapping.values()).index(category)]
    item_indices = np.where(cluster_labels == category_idx)[0]
    recommended_items = np.random.choice(item_indices, num_items, replace=False)
    return recommended_items

# Define label names for Fashion MNIST
fashion_mnist_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

#model to predict the fashion for the style category
def find_style(style):
  recommendations = recommend_items(style, num_items=5)
  im, lable = list(), set()
  for idx in recommendations:
    img, lbl = combined_dataset[idx]
    img = img.numpy().squeeze()
    label_name = fashion_mnist_labels[lbl]
    im.append(img)
    lable.add(label_name)
  return lable, im
