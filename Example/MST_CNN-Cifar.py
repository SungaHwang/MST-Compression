import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from ptflops import get_model_complexity_info
import networkx as nx

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = CNN()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

results_dir = 'results/CNN'
os.makedirs(results_dir, exist_ok=True)
filename = f"{results_dir}/Cifar10_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'CNN_trained_model_weights.pth')

#train_model(net, trainloader, criterion, optimizer)

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def log_experiment_details(model, filename, header="Experiment Settings"):
    with open(filename, 'w') as f:
        f.write(f"{header}:\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
        f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Momentum: {optimizer.param_groups[0]['momentum']}\n")
        f.write(f"Criterion: {criterion.__class__.__name__}\n")
        f.write(f"Batch Size: {trainloader.batch_size}\n\n")

        f.write("Layer Parameters:\n")
        for name, param in model.named_parameters():
            if 'weight' in name:
                f.write(f"{name}: {param.numel()}\n")
        f.write("\n")

log_experiment_details(net, filename)

def prune_and_report_conv_layers_only(model, test_loader, algorithm='kruskal'):
    model.load_state_dict(torch.load('CNN_trained_model_weights.pth'))
    initial_accuracy = evaluate_model(model, test_loader)
    initial_params = sum(p.numel() for p in model.parameters())
    _, initial_flops = get_model_complexity_info(model, (3, 32, 32), as_strings=False)

    total_pruned_params = 0

    with open(filename, 'a') as f:
        f.write("Initial Model Stats:\n")
        f.write(f"Accuracy: {initial_accuracy*100:.2f}%\n")
        f.write(f"Parameters: {initial_params}\n")
        f.write(f"FLOPs: {initial_flops}\n\n")

        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                original_weights = module.weight.data.clone()
                G = nx.Graph()

                for i in range(module.weight.data.size(0)):
                    mask = torch.ones_like(module.weight.data)
                    mask[i] = 0
                    module.weight.data = original_weights * mask
                    
                    pruned_accuracy = evaluate_model(model, test_loader)
                    accuracy_drop = initial_accuracy - pruned_accuracy
                    G.add_node(i, accuracy_drop=accuracy_drop)
                    
                module.weight.data = original_weights.clone()

                for i in range(module.weight.data.size(0)):
                    for j in range(i + 1, module.weight.data.size(0)):
                        edge_weight = abs(G.nodes[i]['accuracy_drop'] - G.nodes[j]['accuracy_drop'])
                        G.add_edge(i, j, weight=edge_weight)

                T = nx.minimum_spanning_tree(G, weight='weight', algorithm=algorithm)
                prune_indices = [node for node, degree in T.degree() if degree == 1]
                mask = torch.ones_like(module.weight.data)
                for idx in prune_indices:
                    mask[idx] = 0

                pruned_weights = original_weights * mask
                module.weight.data = pruned_weights
                module.weight.requires_grad = False  # Disable gradient updates

                remaining_params = pruned_weights.nonzero().size(0)
                total_pruned_params += remaining_params

                # Re-train the model with pruned weights
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
                train_model(model, trainloader, criterion, optimizer, num_epochs=5)

                pruned_accuracy = evaluate_model(model, test_loader)
                _, pruned_flops = get_model_complexity_info(model, (3, 32, 32), as_strings=False)

                f.write(f"{name} Layer - Pruned with {algorithm.capitalize()} algorithm:\n")
                f.write(f"Pruned Accuracy: {pruned_accuracy*100:.2f}%\n")
                f.write(f"Pruned Parameters: {remaining_params}\n")
                f.write(f"Pruned FLOPs: {pruned_flops}\n\n")

        # Final model evaluation
        final_accuracy = evaluate_model(model, test_loader)
        _, final_flops = get_model_complexity_info(model, (3, 32, 32), as_strings=False)

        f.write("After Full Model Pruning - Kruskal Algorithm:\n")
        f.write(f"Final Model Accuracy: {final_accuracy*100:.2f}%\n")
        f.write(f"Final Model Parameters: {total_pruned_params}\n")
        f.write(f"Final Model FLOPs: {final_flops}\n")

prune_and_report_conv_layers_only(net, testloader, 'kruskal')
print("Pruning completed and results are saved to", filename)
