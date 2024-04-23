import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNNModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    correct_original = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_original += (predicted == labels).sum().item()

    accuracy_original = correct_original / total
    print(f'Accuracy on original model: {accuracy_original}')

modified_model = CNNModel()
modified_model.load_state_dict(model.state_dict()) 

sample_images, sample_labels = next(iter(test_loader))

# conv layer
original_feature_maps = modified_model.conv1(sample_images)
modified_feature_maps = modified_model.conv1(sample_images)

accuracies_batch = []
num_features = modified_feature_maps.size(1)

for i in range(num_features):
    modified_feature_maps = original_feature_maps.clone()
    modified_feature_maps[:, i, :, :] = 0  # Remove one element from the ith feature map
    outputs = modified_model.relu(modified_feature_maps)
    outputs = modified_model.pool(outputs)
    outputs = outputs.view(outputs.size(0), -1)
    outputs = modified_model.fc(outputs)

    _, predicted = torch.max(outputs.data, 1)
    correct_modified = (predicted == sample_labels).sum().item()
    accuracy_modified = correct_modified / sample_labels.size(0)

    accuracies_batch.append(accuracy_modified)

for i, acc in enumerate(accuracies_batch):
    print(f'Accuracy on modified input (removed one element from feature map {i}): {acc}')


removal_percentages = list(range(10, 101, 5))
accuracies_percentage = []

for percentage in removal_percentages:
    num_features_to_keep = int(num_features * (1 - percentage / 100))
    sorted_indices = sorted(range(len(accuracies_batch)), key=lambda i: accuracies_batch[i])[:num_features_to_keep]

    new_feature_maps = torch.zeros_like(original_feature_maps)
    for i in sorted_indices:
        new_feature_maps[:, i, :, :] = original_feature_maps[:, i, :, :]

    outputs = modified_model.relu(new_feature_maps)
    outputs = modified_model.pool(outputs)
    outputs = outputs.view(outputs.size(0), -1)
    outputs = modified_model.fc(outputs)

    _, predicted = torch.max(outputs.data, 1)
    correct_modified = (predicted == sample_labels).sum().item()
    new_accuracy = correct_modified / sample_labels.size(0)

    accuracies_percentage.append(new_accuracy)

    print(f'New Accuracy with {percentage}% features: {new_accuracy}')

plt.plot(removal_percentages, accuracies_percentage)
plt.xlabel('Percentage of Features Removed')
plt.ylabel('Accuracy')
plt.title('Accuracy: Percentage of Features Removed')
plt.show()

model_weights = modified_model.state_dict()

for key in model_weights.keys():
    print(f'Layer: {key}, Size: {model_weights[key].size()}')

torch.save(modified_model.state_dict(), 'model_weights_70.pth')