import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime
from ptflops import get_model_complexity_info
import networkx as nx

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

set_seed(42) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 데이터 전처리
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# ResNet18 모델 정의
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR10은 10개 클래스
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

results_dir = 'results/ResNet18'
os.makedirs(results_dir, exist_ok=True)
filename = f"{results_dir}/Cifar10_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), os.path.join(results_dir, 'ResNet18_trained_model_weights.pth'))

#train_model(model, trainloader, criterion, optimizer, num_epochs=10)

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

log_experiment_details(model, filename)

def prune_conv_layers(model, test_loader, algorithm='kruskal'):
    model.load_state_dict(torch.load(os.path.join(results_dir, 'ResNet18_trained_model_weights.pth')))
    initial_accuracy = evaluate_model(model, test_loader)
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _, initial_flops = get_model_complexity_info(model, (3, 224, 224), as_strings=False)

    with open(filename, 'a') as f:
        f.write("Initial Model Stats:\n")
        f.write(f"Accuracy: {initial_accuracy*100:.2f}%\n")
        f.write(f"Parameters: {initial_params}\n")
        f.write(f"FLOPs: {initial_flops}\n\n")

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                original_weights = module.weight.data.clone()
                G = nx.Graph()
                num_filters = module.weight.size(0)

                # Temporarily evaluate model with each filter removed
                for i in range(num_filters):
                    mask = torch.ones(num_filters, dtype=torch.bool)
                    mask[i] = False
                    module.weight.data = original_weights[mask]

                    temp_accuracy = evaluate_model(model, test_loader)
                    accuracy_drop = initial_accuracy - temp_accuracy
                    G.add_node(i, accuracy_drop=accuracy_drop)

                module.weight.data = original_weights  # Restore original weights

                # Build graph based on accuracy drop correlations
                for i in range(num_filters):
                    for j in range(i + 1, num_filters):
                        edge_weight = abs(G.nodes[i]['accuracy_drop'] - G.nodes[j]['accuracy_drop'])
                        G.add_edge(i, j, weight=edge_weight)

                # Determine filters to prune
                T = nx.minimum_spanning_tree(G, weight='weight', algorithm=algorithm)
                prune_indices = [node for node, degree in T.degree() if degree == 1]
                mask = torch.ones(num_filters, dtype=torch.bool)
                mask[prune_indices] = False

                # Apply pruning
                module.weight.data = original_weights[mask]
                module.weight.requires_grad = False  # Freeze pruned filters

                pruned_accuracy = evaluate_model(model, test_loader)
                pruned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                _, pruned_flops = get_model_complexity_info(model, (3, 224, 224), as_strings=False)

                f.write(f"Pruned {name} layer:\n")
                f.write(f"Remaining Accuracy: {pruned_accuracy*100:.2f}%\n")
                f.write(f"Remaining Parameters: {pruned_params}\n")
                f.write(f"Reduced FLOPs: {pruned_flops}\n\n")

        # Final evaluation
        final_accuracy = evaluate_model(model, test_loader)
        final_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _, final_flops = get_model_complexity_info(model, (3, 224, 224), as_strings=False)

        f.write("Final Model Stats After Pruning:\n")
        f.write(f"Accuracy: {final_accuracy*100:.2f}%\n")
        f.write(f"Parameters: {final_params}\n")
        f.write(f"FLOPs: {final_flops}\n")

prune_conv_layers(model, testloader, 'kruskal')

print("Pruning completed and results are saved to", filename)
