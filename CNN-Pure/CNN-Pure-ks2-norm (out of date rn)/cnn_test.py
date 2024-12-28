# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchsummary import summary

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_path = os.path.dirname(__file__)
sub_path = 'Datasets/Images-cnn-test/'
path = os.path.join(script_path, sub_path)

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Using device: {device}')

df = pd.read_csv(path + 'labels.csv')

#df = pd.read_csv('labels.csv')

image_paths = path+df['image'].values
#print(image_paths)
labels = df['label'].values
num_classes = 21

label_mapping = {}
with open(path+'label_mapping.txt', 'r') as file:
    for line in file:
        value, key = line.strip().split(': ')
        label_mapping[int(key)] = value

# Split the data for 70% training, 15% test set, 15% validation set
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.30, random_state=42)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    test_paths, test_labels, test_size=0.5, random_state=42)

def load_images(file_paths, transform, folder_path = path):
    images = []
    for file_path in tqdm(file_paths, desc='Loading images'):
        # Load the image
        with Image.open(os.path.join(folder_path,file_path)) as img:
            # Convert image to RGB if it's not and apply the same basic transformations
            img = img.convert('RGB')
            img = transform(img)
            images.append(img)
    return torch.stack(images)

def calculate_mean_std(stacked_images):
    # Mean and std are calculated across the height and width dimensions (2 and 3)
    mean = stacked_images.view(stacked_images.size(0), stacked_images.size(1), -1).mean(dim=2).mean(dim=0)
    std = stacked_images.view(stacked_images.size(0), stacked_images.size(1), -1).std(dim=2).mean(dim=0)
    return mean, std

basic_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),        # 224x224 is a common "historical" format for benchmarking CNNs
    transforms.ToTensor()
])

train_images = load_images(train_paths, basic_transform)
mean, std = calculate_mean_std(train_images)
normalize_transform = transforms.Normalize(mean=mean, std=std)
train_images = torch.stack([normalize_transform(image) for image in train_images])

# Transformation with normalization for validation and test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),        # 224x224 is a common "historical" format for benchmarking CNNs
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load and normalize the validation and test datasets
val_images = load_images(val_paths, transform)
test_images = load_images(test_paths, transform)

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

# Create TensorDatasets
train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN architecture
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # pooling has no learnable parameters, so we can just use one
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP classifier
        self.fc = nn.Linear(128 * 3 * 3, num_classes)

    def forward(self, x):
        # print("Input size:", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv5(x)))
        # print("Layer size:", x.size())
        x = self.pool(F.relu(self.conv6(x)))
        # print("Layer size:", x.size())
        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor
        # print("Layer size:", x.size())

        # Fully connected layer for classification
        x = self.fc(x)

        return x
    
# Instantiate the basic CNN
model = BasicCNN(num_classes=num_classes).to(device)
summary(model, input_size=(3, 224, 224))

criterion = nn.CrossEntropyLoss()
def train(model, train_loader, path = 'best_model.pth', lr=0.005, num_epochs=5):
  # Define loss function, optimizer and training epochs
  optimizer = optim.AdamW(model.parameters(), lr=lr)


  # Initialize the metrics
  metrics = {'training losses' : [],
             'validation losses' : [],
             'validation accuracies' : [],
             'validation precisions' : [],
             'validation recalls' : [],
             'validation f1 scores' : []}


  # We keep track of the best validation accuracy and save the best model
  best_val_accuracy = 0.0

  # Training loop
  for epoch in range(num_epochs):
      model.train()
      training_loss = 0.0
      for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          training_loss += loss.item()

      # Store average training loss for the epoch
      metrics['training losses'].append(training_loss/len(train_loader))

      # Validation loop
      model.eval()
      validation_loss = 0.0
      true_labels = []
      predicted_labels = []

      with torch.no_grad():
          for images, labels in val_loader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              loss = criterion(outputs, labels)
              validation_loss += loss.item()
              _, predicted = torch.max(outputs.data, 1)

              true_labels.extend(labels.cpu().numpy())
              predicted_labels.extend(predicted.cpu().numpy())

      # Calculate validation loss
      metrics['validation losses'].append(validation_loss / len(val_loader))

      # Calculate metrics
      accuracy = 100*accuracy_score(true_labels, predicted_labels)
      precision = 100*precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
      recall = 100*recall_score(true_labels, predicted_labels, average='macro')
      f1 = 100*f1_score(true_labels, predicted_labels, average='weighted')

      # Store metrics
      metrics['validation accuracies'].append(accuracy)
      metrics['validation precisions'].append(precision)
      metrics['validation recalls'].append(recall)
      metrics['validation f1 scores'].append(f1)

      # save best model based on validation accuracy
      if accuracy > best_val_accuracy:
          best_val_accuracy = accuracy
          torch.save(model.state_dict(), path)
          print(f"Epoch {epoch+1}: Improved validation accuracy to {best_val_accuracy:.2f}%. Model saved.")

      print(f'Epoch {epoch+1}/{num_epochs}', f'Train Loss: {metrics["training losses"][-1]:.4f}, '
          f'Validation Loss: {metrics["validation losses"][-1]:.4f}, '
          f'Accuracy: {metrics["validation accuracies"][-1]:.2f}%')

  return metrics

metrics = train(model, train_loader, path='best_model.pth', num_epochs=10)
model.load_state_dict(torch.load('best_model.pth'))

def visualize(metrics):
  cpu_metrics = {'training losses' : [],
             'validation losses' : [],
             'validation accuracies' : [],
             'validation precisions' : [],
             'validation recalls' : [],
             'validation f1 scores' : []}

  for key, values in metrics.items():
    cpu_metrics[key] = [val.cpu().item() if hasattr(val, 'cpu') else val for val in values]


  losses = ['training losses', 'validation losses']
  other_metrics = ['validation accuracies', 'validation precisions', 'validation recalls', 'validation f1 scores']

  # Create figure and subplots
  fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

  # Plot for losses
  axs[0].set_title('Losses')
  for metric in losses:
      axs[0].plot(cpu_metrics[metric], label=metric)
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Value')
  axs[0].legend()
  axs[0].grid(True)

  # Instantiate a second y-axis sharing the same x-axis
  ax2 = axs[0].twinx()
  ax2.plot(cpu_metrics['validation accuracies'], 'r-', label='Validation Accuracy')
  ax2.set_ylabel('Accuracy (%)')
  ax2.tick_params(axis='y')
  ax2.legend(loc='upper left')

  # Plot for other metrics
  axs[1].set_title('Other Metrics')
  for metric in other_metrics:
      axs[1].plot(cpu_metrics[metric], label=metric)
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Value')
  axs[1].legend()
  axs[1].grid(True)

  plt.tight_layout()
  plt.show()    

def test(model, test_loader):
  # Test loop
  model.eval()
  test_loss = 0.0
  correct = 0
  total = 0
  true_labels = []
  predicted_labels = []
  with torch.no_grad():
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          test_loss += loss.item()
          _, predicted = torch.max(outputs.data, 1)

          true_labels.extend(labels.cpu().numpy())
          predicted_labels.extend(predicted.cpu().numpy())

  # Calculate metrics
  accuracy = 100*accuracy_score(true_labels, predicted_labels)
  precision = 100*precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  recall = 100*recall_score(true_labels, predicted_labels, average='weighted')
  f1 = 100*f1_score(true_labels, predicted_labels, average='weighted')

  # Update progress bar with stats
  print(f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%')

test(model, test_loader)
visualize(metrics)

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP classifier
        self.fc = nn.Linear(256 * 14 * 14, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 14 * 14)  # Flatten the tensor

        x = self.fc(x)

        return x

# deeperCNN = DeeperCNN(num_classes=num_classes).to(device)
# summary(deeperCNN, input_size=(3, 224, 224))
# metrics2 = train(deeperCNN, train_loader, lr=0.0005, num_epochs=10)
# deeperCNN.load_state_dict(torch.load('best_model.pth'))
# print("Loaded best model from:", 'best_model.pth')

# visualize(metrics2)
# test(deeperCNN, test_loader)

class EvenDeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(EvenDeeperCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP classifier
        self.fc = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool(x)

        x = x.view(-1, 512 * 7 * 7)  # Flatten the tensor

        x = self.fc(x)

        return x
    
# even_deeper_CNN = EvenDeeperCNN(num_classes=num_classes).to(device)
# try:
#     summary(even_deeper_CNN.to('cuda'), input_size=(3, 224, 224))
# except:
#     summary(even_deeper_CNN.to('cpu'), input_size=(3, 224, 224))
# # summary(even_deeper_CNN, input_size=(3, 224, 224))
# metrics3 = train(even_deeper_CNN.to(device), train_loader, lr=0.00005, num_epochs=20)
# even_deeper_CNN.load_state_dict(torch.load('best_model.pth'))
# print("Loaded best model from:", 'best_model.pth')