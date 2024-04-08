import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

sample_image, sample_label = next(iter(test_loader))

#conv layer
original_feature_map = modified_model.conv1(sample_image)
modified_feature_map = modified_model.conv1(sample_image)

accuracies = []
num_features = modified_feature_map.size(1)


for i in range(num_features):
    modified_feature_map = original_feature_map.clone()
    modified_feature_map[:, i, :, :] = 0  # Remove one element from the first feature map
    outputs = modified_model.relu(modified_feature_map)
    outputs = modified_model.pool(outputs)
    outputs = outputs.view(outputs.size(0), -1)
    outputs = modified_model.fc(outputs)

    _, predicted = torch.max(outputs.data, 1)
    correct_modified = (predicted == sample_label).sum().item()
    accuracy_modified = correct_modified / sample_label.size(0)

    accuracies.append(accuracy_modified)

for i, acc in enumerate(accuracies):
    print(f'Accuracy on modified input (removed one element{i}): {acc}')


num_features_to_keep = int(num_features * 0.7)
sorted_indices = sorted(range(len(accuracies)), key = lambda i : accuracy_original - accuracies[i], reverse = True)[:num_features_to_keep]

new_features_map = torch.zeros_like(original_feature_map)
for i in sorted_indices:
    new_features_map[:,i,:,:] = original_feature_map[:,i,:,:]

outputs = modified_model.relu(new_features_map)
outputs = modified_model.pool(outputs)
outputs = outputs.view(outputs.size(0), -1)
outputs = modified_model.fc(outputs)

_, predicted = torch.max(outputs.data, 1)
correct_modified = (predicted == sample_label).sum().item()
new_accuracy = correct_modified / sample_label.size(0)

print(f'New Accuracy with 70% features: {new_accuracy}')

model_weights = modified_model.state_dict()

for key in model_weights.keys():
    print(f'Layer: {key}, Size: {model_weights[key].size()}')

torch.save(modified_model.state_dict(), 'model_weights_70.pth')