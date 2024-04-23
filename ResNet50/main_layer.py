import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.models import resnet50
from torch.optim.lr_scheduler import StepLR
import numpy as np
from thop import profile
import networkx as nx
import os
import datetime
import logging
import copy

def setup_logging():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_directory = "ResNet50/log"  # Change directory to ResNet50
    log_filename = f"{current_time}.log"
    
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, log_filename)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Logging setup complete - logging to: " + log_file_path)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(dataset_name):
    transform_options = {
        "MNIST": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        "CIFAR10": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "ImageNet": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    }
    transform = transform_options.get(dataset_name, transform_options["ImageNet"])
    train_set = datasets.ImageFolder(root=f'./data/{dataset_name.lower()}/train', transform=transform)
    test_set = datasets.ImageFolder(root=f'./data/{dataset_name.lower()}/val', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader

def initialize_model():
    model = resnet50(pretrained=False)  # Use ResNet50
    return model

def train_model(model, train_loader, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()}")
        logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

        checkpoint_dir = 'checkpoints/ResNet50'
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth')

    return model

def prune_model_weights(model, target_layers):
    total_pruned_weights = 0 
    for name, param in model.named_parameters():
        if 'conv' in name and name in target_layers:
            weights = param.data.cpu().numpy()
            F, C, H, W = weights.shape
            layer_pruned_weights = 0
            for f in range(F):
                G = nx.Graph()
                for i in range(H):
                    for j in range(W):
                        current_weight = weights[f, :, i, j]
                        if j < W - 1:
                            right_weight = weights[f, :, i, j+1]
                            diff = np.sum(np.abs(current_weight - right_weight))
                            G.add_edge((i, j), (i, j+1), weight=diff)
                        if i < H - 1:
                            down_weight = weights[f, :, i+1, j]
                            diff = np.sum(np.abs(current_weight - down_weight))
                            G.add_edge((i, j), (i+1, j), weight=diff)
                T = nx.minimum_spanning_tree(G, algorithm='kruskal')
                leaves = [node for node, degree in dict(T.degree()).items() if degree == 1]
                for leaf in leaves:
                    i, j = leaf
                    weights[f, :, i, j] = 0
            logging.info(f"{name}: Pruned {layer_pruned_weights} weights.")
            total_pruned_weights += layer_pruned_weights
            param.data = torch.from_numpy(weights).to(param.device)
    logging.info(f"Total pruned weights in the model: {total_pruned_weights}")
    return model

def prune_model_filters_by_importance(model, train_loader, test_loader, target_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_pruned_filters = 0
    for name, param in model.named_parameters():
        if 'conv' in name and name in target_layers:
            importance_scores = compute_layer_importance(model, train_loader, test_loader, name)
            G = nx.Graph()
            num_filters = len(importance_scores)
            for i in range(num_filters):
                for j in range(i + 1, num_filters):
                    G.add_edge(i, j, weight=np.abs(importance_scores[i] - importance_scores[j]))
            T = nx.minimum_spanning_tree(G, algorithm='kruskal')
            leaves = [node for node, degree in dict(T.degree()).items() if degree == 1]
            num_pruned_filters = len(leaves) 
            total_pruned_filters += num_pruned_filters 

            logging.info(f"{name}: Pruned {num_pruned_filters} filters.")
            for leaf in leaves:
                param.data[leaf, :, :, :] = 0
    logging.info(f"Total pruned filters in the model: {total_pruned_filters}")
    return model

def compute_layer_importance(model, train_loader, test_loader, layer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    original_accuracy = evaluate_model(model, test_loader)
    importance_scores = []

    for name, param in model.named_parameters():
        if name == layer_name:
            for i in range(param.shape[0]):
                original_weight = param.data[i].clone()
                param.data[i] = 0
                reduced_accuracy = evaluate_model(model, test_loader)
                importance_score = original_accuracy - reduced_accuracy 
                importance_scores.append(importance_score)
                param.data[i] = original_weight

    return importance_scores

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

import time

def compute_flops(model, input_size=(1, 3, 224, 224), device = 'cuda'):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return flops

def count_nonzero_parameters(model):
    nonzero_count = sum(p.numel() for p in model.parameters() if p.data.ne(0).sum())
    return nonzero_count

def evaluate_model_full(model, test_loader, device):
    model.to(device)
    model.eval()
    # Accuracy
    correct = 0
    total = 0
    # Inference Time
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = end_time - start_time

    # FLOPs
    flops = compute_flops(model)

    # Non-zero Parameters
    nonzero_params = count_nonzero_parameters(model)

    return accuracy, inference_time, flops, nonzero_params


def main():
    setup_logging()
    logging.info("Starting the program")

    seed = 0
    set_seed(seed)

    # CIFAR10 is not really the typical dataset for ResNet50 due to input size expectations, but for simplicity we'll use it here.
    train_loader, test_loader = load_data("CIFAR10")
    model = initialize_model()

    # Train the model
    trained_model = train_model(model, train_loader, epochs=10)  # Reduced epochs for quick testing
    original_model = copy.deepcopy(trained_model)

    # Define target layers for pruning in ResNet50: Select first convolution in each bottleneck
    target_layers = [
        'layer1.0.conv1', 'layer1.1.conv1', 'layer1.2.conv1',
        'layer2.0.conv1', 'layer2.1.conv1', 'layer2.2.conv1', 'layer2.3.conv1',
        'layer3.0.conv1', 'layer3.1.conv1', 'layer3.2.conv1', 'layer3.3.conv1', 'layer3.4.conv1', 'layer3.5.conv1',
        'layer4.0.conv1', 'layer4.1.conv1', 'layer4.2.conv1'
    ]

    # Prune model weights
    pruned_model_weights = prune_model_weights(copy.deepcopy(original_model), target_layers)
    results = {
        'original': evaluate_model_full(original_model, test_loader, "cuda"),
        'pruned_weights': evaluate_model_full(pruned_model_weights, test_loader, "cuda")
    }

    # Prune model filters by importance
    pruned_model_filters = prune_model_filters_by_importance(copy.deepcopy(original_model), train_loader, test_loader, target_layers)
    results['pruned_filters'] = evaluate_model_full(pruned_model_filters, test_loader, "cuda")

    # Display results
    print(f"Original Model: Accuracy: {results['original'][0]}, Inference Time: {results['original'][1]}, FLOPs: {results['original'][2]}, Non-zero Params: {results['original'][3]}")
    logging.info(f"Original Model: Accuracy: {results['original'][0]}, Inference Time: {results['original'][1]}, FLOPs: {results['original'][2]}, Non-zero Params: {results['original'][3]}")
    print(f"Pruned by Weights: Accuracy: {results['pruned_weights'][0]}, Inference Time: {results['pruned_weights'][1]}, FLOPs: {results['pruned_weights'][2]}, Non-zero Params: {results['pruned_weights'][3]}")
    logging.info(f"Pruned by Weights: Accuracy: {results['pruned_weights'][0]}, Inference Time: {results['pruned_weights'][1]}, FLOPs: {results['pruned_weights'][2]}, Non-zero Params: {results['pruned_weights'][3]}")
    print(f"Pruned by Filters: Accuracy: {results['pruned_filters'][0]}, Inference Time: {results['pruned_filters'][1]}, FLOPs: {results['pruned_filters'][2]}, Non-zero Params: {results['pruned_filters'][3]}")
    logging.info(f"Pruned by Filters: Accuracy: {results['pruned_filters'][0]}, Inference Time: {results['pruned_filters'][1]}, FLOPs: {results['pruned_filters'][2]}, Non-zero Params: {results['pruned_filters'][3]}")

if __name__ == "__main__":
    main()
