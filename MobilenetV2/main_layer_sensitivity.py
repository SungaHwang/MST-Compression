import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import networkx as nx
import os
import datetime
import logging
import copy
import random
import argparse
import matplotlib.pyplot as plt
import json
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로깅 설정
def setup_logging():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_directory = "MobileNetV2/log_sen"
    log_filename = f"{current_time}.log"

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Logging setup complete - logging to: " + log_file_path)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.cuda.manual_seed_all(seed)

# 데이터 로딩
def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = datasets.ImageFolder(root='./data/train', transform=transform)
        test_set = datasets.ImageFolder(root='./data/val', transform=transform)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    logging.info("Dataset: %s", dataset_name)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 모델 정의 및 초기화
def initialize_model():
    model = models.mobilenet_v2(pretrained=False)
    model.to(device)
    return model

# 모델 학습
def train_model(model, train_loader, epochs=100, lr=0.01, momentum=0.9, patience=20):
    device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    best_val_accuracy = 0
    no_improvement_count = 0  # For early stopping
    
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        epoch_accuracy = total_correct / total_samples
        logging.info(f"Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {epoch_accuracy}")

        # Check for early stopping based on validation accuracy
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                logging.info(f"No improvement in validation accuracy for {patience} epochs. Early stopping...")
                break

        scheduler.step()

    return model

# 프루닝 방법 1: 가중치에 기반한 프루닝
def prune_model_weights(model, target_layers, pruning_percent=10, algorithm='kruskal'):
    model = model.to(device)
    total_pruned_weights = 0
    for name, param in model.named_parameters():
        if name in target_layers:
            logging.info(f"Pruning layer: {name}")
            weights = param.data.cpu().numpy()  # 가중치를 CPU로 이동 후 NumPy 배열로 변환
            F, C, H, W = weights.shape
            layer_pruned_weights = 0

            # 각 필터에 대해 MST를 만들기 위한 그래프 생성 및 가중치 연관성 추가
            for f in range(F):
                G = nx.Graph()
                for c in range(C):
                    for i in range(H):
                        for j in range(W):
                            G.add_node((c, i, j))  # 각 가중치를 노드로 추가
                            current_weight = weights[f, c, i, j]
                            if j < W - 1:
                                right_weight = weights[f, c, i, j + 1]
                                diff = np.abs(current_weight - right_weight)
                                G.add_edge((c, i, j), (c, i, j + 1), weight=diff)
                            if i < H - 1:
                                down_weight = weights[f, c, i + 1, j]
                                diff = np.abs(current_weight - down_weight)
                                G.add_edge((c, i, j), (c, i + 1, j), weight=diff)
                            if i < H - 1 and j < W - 1:
                                diag_weight = weights[f, c, i + 1, j + 1]
                                diff = np.abs(current_weight - diag_weight)
                                G.add_edge((c, i, j), (c, i + 1, j + 1), weight=diff)

                # MST 계산
                T = nx.minimum_spanning_tree(G, algorithm=algorithm)

                # 제거해야 할 노드 개수 계산
                num_nodes_to_prune = int(G.number_of_nodes() * (pruning_percent / 100))

                # 제거 대상 노드 선택
                nodes_to_prune = set()
                for edge in reversed(list(T.edges(data=True))):  # MST에서 역순으로 엣지 순회
                    if len(nodes_to_prune) < num_nodes_to_prune:
                        nodes_to_prune.add(edge[0])
                        #nodes_to_prune.add(edge[1])
                    else:
                        break

                # 가중치 제거
                for node in nodes_to_prune:
                    c, i, j = node
                    weights[f, c, i, j] = 0
                    layer_pruned_weights += 1

            logging.info(f"{name}: Pruned {layer_pruned_weights} / {F * C * H * W} weights ({pruning_percent}%)")
            total_pruned_weights += layer_pruned_weights

            # 프루닝된 가중치를 모델에 적용
            param.data = torch.from_numpy(weights).to(param.device)

    logging.info(f"Total pruned weights: {total_pruned_weights}")
    return model

# 프루닝 방법 2: 필터 중요도에 기반한 프루닝
def prune_model_filters_by_importance(model, train_loader, test_loader, target_layers, pruning_percent=10, algorithm='kruskal'):
    model = model.to(device)
    total_pruned_filters = 0
    for name, param in model.named_parameters():
        if name in target_layers:
            logging.info(f"Pruning layer: {name}")
            importance_scores = compute_layer_importance(model, train_loader, test_loader, name)
            num_filters = len(importance_scores)

            G = nx.Graph()
            for i in range(num_filters):
                for j in range(i + 1, num_filters):
                    G.add_edge(i, j, weight=-np.abs(importance_scores[i] - importance_scores[j]))

            mst = nx.minimum_spanning_tree(G, weight='weight', algorithm=algorithm)
            sorted_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            
            num_filters_to_prune = int(num_filters * pruning_percent / 100)
            pruned_filters = set()

            # Remove edges and count unique filters until we reach the target number
            for edge in sorted_edges[::-1]:
                if len(pruned_filters) < num_filters_to_prune:
                    pruned_filters.add(edge[1])
                if len(pruned_filters) >= num_filters_to_prune:
                    break

            pruned_filters = list(pruned_filters)
            mask = torch.ones(num_filters, dtype=torch.float32, device=device)
            mask[pruned_filters] = 0
            param.data *= mask[:, None, None, None]

            logging.info(f"{name}: Pruned {len(pruned_filters)} filters at {pruning_percent}%.")
            total_pruned_filters += len(pruned_filters)

    logging.info(f"Total pruned filters in the model: {total_pruned_filters}")
    return model

def compute_layer_importance(model, train_loader, test_loader, layer_name):
    model = model.to(device)
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

# 모델 평가
def evaluate_model(model, test_loader):
    model = model.to(device)
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
    return accuracy

def main(args):
    print(device)
    setup_logging()
    logging.info("Starting the program")
    logging.info("Arguments: %s", args)

    seed = 0
    set_seed(seed)

    train_loader, test_loader = load_data(args.dataset)

    # 원본 모델 초기화
    model = initialize_model()

    # 저장된 모델 경로 설정
    save_directory = 'saved_models/MobileNetV2'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    filename = f"original_model_{args.dataset}.pth"
    save_path = os.path.join(save_directory, filename)

    # 첫 번째 실행 시에만 모델을 훈련하고 저장
    if not os.path.exists(save_path):
        trained_model = train_model(model, train_loader, epochs=args.epochs)
        torch.save(trained_model.state_dict(), save_path)
        print(f"Original model weights saved to {save_path}")
    else:
        # 저장된 모델 불러오기
        trained_model = initialize_model()
        trained_model.load_state_dict(torch.load(save_path))
        trained_model.eval()
        print(f"Original model weights loaded from {save_path}")

    pruning_results = {}

    for target_layer in args.target_layers:
        # 각 레이어마다 프루닝 정도에 따른 정확도를 저장할 딕셔너리 초기화
        layer_results = {}

        for pruning_percent in range(0, 101, 10):
            # 각 프루닝 정도에 대해 모델을 새로 초기화
            model = copy.deepcopy(trained_model)
            model.to(device)

            # 해당 레이어에 대한 프루닝 수행
            if args.prune_method == 'weights':
                pruned_model = prune_model_weights(model, target_layers=[target_layer],
                                                   pruning_percent=pruning_percent, algorithm=args.algorithm)
            elif args.prune_method == 'filters':
                pruned_model = prune_model_filters_by_importance(model, train_loader, test_loader,
                                                                 target_layers=[target_layer],
                                                                  pruning_percent=pruning_percent,
                                                                 algorithm=args.algorithm)
            else:
                raise ValueError("Unsupported pruning method: {}".format(args.prune_method))

            # 프루닝된 모델의 정확도 측정
            accuracy = evaluate_model(pruned_model, test_loader)

            # 결과 딕셔너리에 정확도 저장
            layer_results[f"{target_layer}_{pruning_percent}%"] = accuracy

        # 각 레이어의 프루닝 결과를 전체 결과 딕셔너리에 저장
        pruning_results[target_layer] = layer_results

    # 결과 딕셔너리 저장
    file_name = f"MobileNetV2: pruning_results_{args.algorithm}_{args.dataset}_{args.prune_method}.json"
    with open(file_name, 'w') as f:
        json.dump(pruning_results, f)

    # 그래프 그리기
    plot_pruning_results(pruning_results, args.algorithm, args.dataset, args.prune_method)

def generate_n_colors(n):
    # Generate n distinct colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(n)]
    return colors

def plot_pruning_results(pruning_results, algorithm, dataset, prune_method):
    plt.figure(figsize=(8, 6))
    layers = list(pruning_results.keys())
    num_layers = len(layers)
    colors = generate_n_colors(num_layers)

    for i, (layer_name, layer_results) in enumerate(pruning_results.items()):
        pruning_percentages = [int(p.split('_')[-1].replace('%', '')) for p in layer_results.keys()]
        accuracies = [layer_results[p] for p in layer_results.keys()]
        color = colors[i % num_layers]
        plt.plot(pruning_percentages, accuracies, label=layer_name, marker='o', linestyle='-', color=color)

    plt.title(f"MobileNetV2: Pruning Accuracy by Layer and Pruning Percentage ({algorithm})")
    plt.xlabel("Pruning Percentage")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 101, 10))
    plt.tight_layout()
    plt.savefig(f"MobilenetV2:Pruning_Accuracy_{dataset}_{prune_method}_{algorithm}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pruning")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "ImageNet"],
                        help="Dataset for pruning")
    parser.add_argument("--algorithm", type=str, default="kruskal", choices=["kruskal", "prim"],
                        help="Algorithm for pruning")
    parser.add_argument("--target_layers", nargs="+", default=[
        'features.1.conv.0.0.weight',
        'features.1.conv.1.weight',
        'features.2.conv.0.0.weight',
        'features.2.conv.1.0.weight',
        'features.2.conv.2.weight',
        'features.3.conv.0.0.weight',
        'features.3.conv.1.0.weight',
        'features.3.conv.2.weight',
        'features.4.conv.0.0.weight',
        'features.4.conv.1.0.weight',
        'features.4.conv.2.weight',
        'features.5.conv.0.0.weight',
        'features.5.conv.1.0.weight',
        'features.5.conv.2.weight',
        'features.6.conv.0.0.weight',
        'features.6.conv.1.0.weight',
        'features.6.conv.2.weight',
        'features.7.conv.0.0.weight',
        'features.7.conv.1.0.weight',
        'features.7.conv.2.weight',
        'features.8.conv.0.0.weight',
        'features.8.conv.1.0.weight',
        'features.8.conv.2.weight',
        'features.9.conv.0.0.weight',
        'features.9.conv.1.0.weight',
        'features.9.conv.2.weight',
        'features.10.conv.0.0.weight',
        'features.10.conv.1.0.weight',
        'features.10.conv.2.weight',
        'features.11.conv.0.0.weight',
        'features.11.conv.1.0.weight',
        'features.11.conv.2.weight',
        'features.12.conv.0.0.weight',
        'features.12.conv.1.0.weight',
        'features.12.conv.2.weight',
        'features.13.conv.0.0.weight',
        'features.13.conv.1.0.weight',
        'features.13.conv.2.weight',
        'features.14.conv.0.0.weight',
        'features.14.conv.1.0.weight',
        'features.14.conv.2.weight',
        'features.15.conv.0.0.weight',
        'features.15.conv.1.0.weight',
        'features.15.conv.2.weight',
        'features.16.conv.0.0.weight',
        'features.16.conv.1.0.weight',
        'features.16.conv.2.weight',
        'features.17.conv.0.0.weight',
        'features.17.conv.1.0.weight',
        'features.17.conv.2.weight'
    ], help="Target layers for pruning")
    parser.add_argument("--prune_method", type=str, default="filters", choices=["weights", "filters"],
                        help="Pruning method: 'weights' or 'filters'")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()
    main(args)
