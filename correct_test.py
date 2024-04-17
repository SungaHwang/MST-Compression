import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(0)

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = CNNModel()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train(model, device, train_loader, optimizer, criterion, epochs=5)

torch.save(model.state_dict(), 'original_CNN.pth')

def compare_models(model1, model2, device, description1="Original Model", description2="Kruskal Model"):
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    layer_details = {}

    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        total_weights = p1.numel()
        zero_count = (p1 == 0).sum().item()
        zero_count_kruskal = (p2 == 0).sum().item()
        layer_details[name1] = {'total_weights': total_weights, 'zero_weights_original': zero_count, 'zero_weights_kruskal': zero_count_kruskal}
        print(f"{name1}: Total weights={total_weights}, Zero weights (Original)={zero_count}, Zero weights (Kruskal)={zero_count_kruskal}")


model_original = CNNModel()
model_kruskal = CNNModel()
model_original.load_state_dict(torch.load('original_CNN.pth', map_location=device))
model_kruskal.load_state_dict(torch.load('new_model/kruskal_weights.pth', map_location=device))


compare_models(model_original, model_kruskal, device)
