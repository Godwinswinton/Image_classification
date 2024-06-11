#import the required library 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from tqdm import tqdm
import json

#This is to set a device where the code needs to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Image path of the celebrities
image_folder_path = 'archive/img_align_celeba/img_align_celeba'
#Preprocessed labled data.
label_path = 'attribute.csv'
#Evaluation path
eval_path = '/home/swinton/Documents/PROJECT/archive/list_eval_partition.csv'

#Convert image in particular format
image_size = (64, 64)
data_transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])

#Read CSV to get lables
eval_list = pd.read_csv(eval_path)['partition'].values  
eval_name = pd.read_csv(eval_path)['image_id'].values
labels = pd.read_csv(label_path).values

#Hyperparemeter tuning
parameters = {
    'batch_size': [64, 128],
    'learning_rate': [0.01, 0.001],
    'num_features': [512, 1024],
    'drop_out': [0.2, 0.5]
}

#Funtion to find the lable and its weights
def generate_class_weights(class_series, class_labels):
    mlb = None
    n_samples = len(class_series)
    n_classes = len(class_series[0])

    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))

#transform image and lables into tensors
class CelebADataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.classes = list(pd.read_csv(label_path).columns)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        label = torch.Tensor(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'label': label}
        return sample

indx, indy, recall = [0]*3, [0]*3, 0
for i in eval_list:
    if recall == i - 1:
        recall = i
        indy[recall] += indy[recall - 1] + 1
        indx[recall] = indy[recall]
    else:
        indy[recall] += 1

train_list = [os.path.join(image_folder_path, name) for name in eval_name[indx[0]:50000]]
train_label_list = labels[indx[0]:50000]
val_list = [os.path.join(image_folder_path, name) for name in eval_name[indx[1]:indx[1] + 5000]]
val_label_list = labels[indx[1]:(indx[1] + 5000)]
test_list = [os.path.join(image_folder_path, name) for name in eval_name[indx[2]:indx[2] + 5000]]
test_label_list = labels[indx[2]:(indx[2] + 5000)]

#Split data into train and test split
train_dataset = CelebADataset(train_list, train_label_list, data_transform)
val_dataset = CelebADataset(val_list, val_label_list, data_transform)
test_dataset = CelebADataset(test_list, test_label_list, data_transform)

def CelebA_DataLoader(batch_size, device):
    num_workers = 0 if device.type == 'cuda' else 2
    pin_memory = True if device.type == 'cuda' else False
    classes = train_dataset.classes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class_weights = generate_class_weights(train_label_list, train_dataset.classes)

#Hamming_score is to calculate the accuracy of the model
def hamming_score(y_true, y_pred):
    num_samples = len(y_true)
    total_correct = 0
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        correct_labels = (true_labels == pred_labels).sum()
        total_correct += correct_labels
    hamming_score = total_correct / (num_samples * len(y_true[0]))
    return hamming_score

def get_multilabel_evaluation(model, test_loader):
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for dir_ in test_loader:
            inputs, targets = dir_.values()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.01).float()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return all_predictions, all_targets

#Train model function is used to train the model 
def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    train_losses = []
    train_hamming_scores = []
    val_losses = []
    val_hamming_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train_loss = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for dict_ in train_bar:
            inputs, labels = dict_.values()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            predicted_labels = (outputs > 0.5).float()
            hamming_score_value = hamming_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            train_bar.set_postfix(loss=loss.item(), hamming_score=hamming_score_value)
            
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_hamming_scores.append(hamming_score_value)

        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for dict_ in val_loader:
                inputs, labels = dict_.values()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * inputs.size(0)

                predicted_labels = (outputs > 0.5).float()
                val_hamming_score = hamming_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())

        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_hamming_scores.append(val_hamming_score)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Hamming Score: {hamming_score_value:.4f}, Val Loss: {val_loss:.4f}, Val Hamming Score: {val_hamming_score:.4f}")
    
    return [train_losses, train_hamming_scores, val_losses, val_hamming_scores]

#Define model funtion is to set the architecture of the model and define other function like optimizer, loss function
def define_model(train_loader, val_loader, learning_rate, num_features, dropout_prob, device, test_loader):
    Model = resnet50(weights=ResNet50_Weights.DEFAULT)
    Model.fc = nn.Sequential(
        nn.Linear(Model.fc.in_features, num_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_prob),
        nn.Linear(num_features, len(train_dataset.classes))
    )
    Model = Model.to(device)
    weight_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)    
    optimizer = optim.Adam(Model.parameters(), lr=learning_rate)
    Metric = []
    Metric.append(train_model(Model, device, train_loader, val_loader, criterion, optimizer, num_epochs=10))

    true, pred = get_multilabel_evaluation(Model, test_loader) 
    predictions_np = np.concatenate(pred)
    targets_np = np.concatenate(true)
    predictions_binary = predictions_np > 0.5
    precision, recall, f_score, _ = precision_recall_fscore_support(targets_np, predictions_binary, average='weighted')
    Metric.append([precision, recall, f_score])
    return Model, Metric


Metric = {}

for batch_size in parameters['batch_size']:
    train_loader, val_loader, test_loader = CelebA_DataLoader(batch_size, device)
    for learning_rate in parameters['learning_rate']:
        for num_features in parameters['num_features']:
            for drop_out in parameters['drop_out']:
                name_ = 'CelebA' + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(num_features) + '_' + str(drop_out) + '.pth'
                print(f'Model: | batch_size = {batch_size} learning_rate = {learning_rate}, num_features = {num_features}, drop_out = {drop_out} |')
                Model, Metric[name_] = define_model(train_loader, val_loader, learning_rate, num_features, drop_out, device, test_loader)
                torch.save(Model.state_dict(), name_)

#collect the meteric of the trained model
file_path = "data.json"

with open(file_path, "w") as json_file:
    json.dump(Metric, json_file)

print("Data saved to", file_path)