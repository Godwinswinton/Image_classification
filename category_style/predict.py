import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the model which gave high accuracy in the model evaluation
def model_load(num_features, dropout_prob, device, name):
    Model = resnet50(weights=ResNet50_Weights.DEFAULT)
    Model.fc = nn.Sequential(
        nn.Linear(Model.fc.in_features, num_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_prob),
        nn.Linear(num_features, 3)
    )
    Model = Model.to(device)
    Model.load_state_dict(torch.load(name))
    return Model

Model = model_load(1024, 0.2, device, 'category_style/model/CelebA_64_0.01_1024_0.2.pth') #input the model from the model folder

#Convert images into vectors
image_size = (64, 64)
data_transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])



category_lables = ["Casual Appearance","Formal Appearance","Intellectual Appearance"]

#predict model
def predict(image_path):
    Model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = data_transform(image)
    image_tensor = image_tensor.unsqueeze(0)  
    image_tensor = image_tensor.to(device) 
    with torch.no_grad():
        output = Model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_classes = torch.zeros_like(probabilities, dtype=torch.int)
    threshold = 0.3
    predicted_classes[probabilities >= threshold] = 1
    predicted = predicted_classes[0].cpu().tolist()
    label_name = category_lables[predicted.index(1)]
    return label_name