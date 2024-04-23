import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import numpy as np
from thop import profile
import networkx as nx
import os
import datetime
import logging

# 로깅 설정
def setup_logging():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_directory = "log"
    log_filename = f"{current_time}.log"
    
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Logging setup complete - logging to: " + log_file_path)

# 데이터 로딩
def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:  # ImageNet
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        train_set = datasets.ImageFolder(root='./data/train', transform=transform)
        test_set = datasets.ImageFolder(root='./data/val', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 모델 정의 및 초기화
def initialize_model():
    model = resnet18(pretrained=False)
    return model

# 모델 학습
def train_model(model, train_loader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

        checkpoint_dir = 'checkpoints/ResNet18'
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth')

    return model

# 1
def prune_model_weights(model, algorithm='prim'):
    for name, param in model.named_parameters():
        if 'conv' in name and len(param.shape) == 4:
            G = nx.Graph()
            weights = param.data.cpu().numpy()
            for i in range(weights.shape[0]):
                for j in range(i + 1, weights.shape[0]):
                    weight_diff = np.abs(weights[i] - weights[j]).sum()
                    G.add_edge(i, j, weight=weight_diff)
            
            if algorithm == 'kruskal':
                T = nx.minimum_spanning_tree(G, algorithm='kruskal')
                logging.info(f"Using Kruskal's algorithm for weight MST in layer {name}")
            elif algorithm == 'prim':
                T = nx.minimum_spanning_tree(G, algorithm='prim')
                logging.info(f"Using Prim's algorithm for MST weight in layer {name}")
            
            leaves = [node for node, degree in dict(T.degree()).items() if degree == 1]
            for leaf in leaves:
                weights[leaf] = 0
            param.data = torch.from_numpy(weights).to(param.device)

    return model


# 2
def compute_filter_importance(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    original_accuracy = evaluate_model(model, test_loader)
    importance_scores = []

    for name, param in model.named_parameters():
        if 'conv' in name and len(param.shape) == 4:
            for i in range(param.shape[0]):
                original_weight = param.data[i].clone()
                param.data[i] = 0
                reduced_accuracy = evaluate_model(model, test_loader)
                importance_score = original_accuracy - reduced_accuracy 
                importance_scores.append(importance_score)
                param.data[i] = original_weight
    return importance_scores

def prune_model_filters_by_importance(model, train_loader, test_loader, algorithm='prim'):
    importance_scores = compute_filter_importance(model, train_loader, test_loader)
    G = nx.Graph()
    for i in range(len(importance_scores)):
        for j in range(i + 1, len(importance_scores)):
            G.add_edge(i, j, weight=np.abs(importance_scores[i] - importance_scores[j]))
    
    if algorithm == 'kruskal':
        T = nx.minimum_spanning_tree(G, algorithm='kruskal')
        logging.info(f"Using kruskal's algorithm for MST filter")

    elif algorithm == 'prim':
        T = nx.minimum_spanning_tree(G, algorithm='prim')
        logging.info(f"Using kruskal's algorithm for MST filter")

    leaves = [node for node, degree in dict(T.degree()).items() if degree == 1]
    for leaf in leaves:
        list(model.parameters())[0][leaf] = 0

    return model


# 성능 평가
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

def compute_flops(model, input_size=(1, 3, 224, 224)):
    model.eval()
    input = torch.randn(input_size)
    flops, params = profile(model, inputs=(input,), verbose=False)
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
    setup_logging()  # 로깅 설정 호출
    logging.info("Starting the program")

    train_loader, test_loader = load_data("CIFAR10")
    model = initialize_model()
    trained_model = train_model(model, train_loader)

    # 평가지표 저장
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 원본 모델 평가
    results['original'] = evaluate_model_full(trained_model, test_loader, device)
    
    # 첫 번째 경량화 방법 적용 후 평가
    pruned_model_weights = prune_model_weights(trained_model)
    results['pruned_weights'] = evaluate_model_full(pruned_model_weights, test_loader, device)

    # 두 번째 경량화 방법 적용 후 평가
    pruned_model_filters = prune_model_filters_by_importance(trained_model, train_loader, test_loader)
    results['pruned_filters'] = evaluate_model_full(pruned_model_filters, test_loader, device)

    # 결과 출력
    print(f"Original Model: Accuracy: {results['original'][0]}, Inference Time: {results['original'][1]}, FLOPs: {results['original'][2]}, Non-zero Params: {results['original'][3]}")
    logging.info(f"Original Model: Accuracy: {results['original'][0]}, Inference Time: {results['original'][1]}, FLOPs: {results['original'][2]}, Non-zero Params: {results['original'][3]}")
    print(f"Pruned by Weights: Accuracy: {results['pruned_weights'][0]}, Inference Time: {results['pruned_weights'][1]}, FLOPs: {results['pruned_weights'][2]}, Non-zero Params: {results['pruned_weights'][3]}")
    logging.info(f"Pruned by Weights: Accuracy: {results['pruned_weights'][0]}, Inference Time: {results['pruned_weights'][1]}, FLOPs: {results['pruned_weights'][2]}, Non-zero Params: {results['pruned_weights'][3]}")
    print(f"Pruned by Filters: Accuracy: {results['pruned_filters'][0]}, Inference Time: {results['pruned_filters'][1]}, FLOPs: {results['pruned_filters'][2]}, Non-zero Params: {results['pruned_filters'][3]}")
    logging.info(f"Pruned by Filters: Accuracy: {results['pruned_filters'][0]}, Inference Time: {results['pruned_filters'][1]}, FLOPs: {results['pruned_filters'][2]}, Non-zero Params: {results['pruned_filters'][3]}")

if __name__ == "__main__":
    main()