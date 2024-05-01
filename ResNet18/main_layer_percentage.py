import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import torchvision.models as models
import numpy as np
from thop import profile
import networkx as nx
import os
import datetime
import logging
import copy
import random
import argparse


# 로깅 설정
def setup_logging():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_directory = "ResNet18/log"
    log_filename = f"{current_time}.log"
    
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Logging setup complete - logging to: " + log_file_path)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 데이터 로딩
def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "ImageNet":
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
    model = resnet18(pretrained=False)
    return model


# 모델 학습
def train_model(model, train_loader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
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
# 프루닝 방법 1: 가중치에 기반한 프루닝
def prune_model_weights(model, target_layers, pruning_percent=10, algorithm='kruskal'):
    total_pruned_weights = 0
    for name, param in model.named_parameters():
        if name in target_layers:
            logging.info(f"Pruning layer: {name}")
            pruning_percent = pruning_percents[name]  # 각 레이어별 프루닝 비율을 딕셔너리에서 가져옴
            logging.info(f"Pruning layer: {name} at {pruning_percent}%")

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
                        nodes_to_prune.add(edge[1])
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

# 2
# 프루닝 방법 2: 필터 중요도에 기반한 프루닝
def prune_model_filters_by_importance(model, train_loader, test_loader, target_layers, pruning_percents= 10, algorithm='kruskal'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
            
            num_filters_to_prune = int(num_filters * pruning_percents[name] / 100)
            pruned_filters = set()

            # Remove edges and count unique filters until we reach the target number
            for edge in sorted_edges[::-1]:
                if len(pruned_filters) < num_filters_to_prune:
                    pruned_filters.update(edge[:2])
                if len(pruned_filters) >= num_filters_to_prune:
                    break

            pruned_filters = list(pruned_filters)
            mask = torch.ones(num_filters, dtype=torch.float32, device=device)
            mask[pruned_filters] = 0
            param.data *= mask[:, None, None, None]

            logging.info(f"{name}: Pruned {len(pruned_filters)} filters at {pruning_percents}%.")
            total_pruned_filters += len(pruned_filters)

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

def compute_flops(model, input_size=(1, 3, 224, 224), device = 'cuda'):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = torch.randn(input_size).to(device)

    total_flops = 0  # FLOPs를 저장할 변수 초기화

    def conv_flops_hook(module, input, output):
        # 0이 아닌 가중치만 계산
        weight = module.weight
        active_elements_count = torch.sum(weight != 0).item()
        
        # Convolution FLOPs 계산
        output_dims = output.shape[2:]  # output height and width
        kernel_dims = module.kernel_size  # kernel height and width
        in_channels = module.in_channels / module.groups
        flops = torch.prod(torch.tensor(kernel_dims)).item() * in_channels * active_elements_count * torch.prod(torch.tensor(output_dims)).item()

        nonlocal total_flops
        total_flops += flops  # 계산된 FLOPs를 전체에 누적

    hooks = []

    # 모든 Conv2d 모듈에 훅 등록
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(conv_flops_hook)
            hooks.append(hook)

    # 더미 입력으로 모델 실행하여 훅 트리거
    with torch.no_grad():
        model(input)

    # 훅 제거
    for hook in hooks:
        hook.remove()

    return total_flops

def count_nonzero_parameters(model):
    total_params = 0
    zero_count = 0
    for param in model.parameters():
        param_count = param.numel() 
        zero_count += torch.sum(param == 0).item()  
        total_params += param_count 

    nonzero_count = total_params - zero_count
    return nonzero_count


def evaluate_model_full(model, test_loader, device):
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
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

# 프루닝 비율 딕셔너리 정의
pruning_percents = {
    'conv1.weight' : 0,
    'layer1.0.conv1.weight': 0,
    'layer1.0.conv2.weight': 0,
    'layer1.1.conv1.weight': 0,
    'layer1.1.conv2.weight': 0,
    'layer2.0.conv1.weight': 0,
    'layer2.0.conv2.weight': 0,
    'layer2.1.conv1.weight': 0,
    'layer2.1.conv2.weight': 0,
    'layer3.0.conv1.weight': 10,
    'layer3.0.conv2.weight': 10,
    'layer3.1.conv1.weight': 30,
    'layer3.1.conv2.weight': 30,
    'layer4.0.conv1.weight': 10,
    'layer4.0.conv2.weight': 10,
    'layer4.1.conv1.weight': 30,
    'layer4.1.conv2.weight': 30
}


def main(args):
    setup_logging()
    logging.info("Starting the program")
    logging.info("Arguments: %s", args)


    seed = 0
    set_seed(seed)

    train_loader, test_loader = load_data(args.dataset)
    model = initialize_model()

    trained_model = train_model(model, train_loader, epochs=args.epochs)

    original_model = copy.deepcopy(trained_model)

    # 평가지표 저장
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 원본 모델 평가
    original_model = copy.deepcopy(trained_model)
    results['original'] = evaluate_model_full(original_model, test_loader, device)

    # 프루닝 실행
    if args.pruning_method == "prune_model_weights":
        pruned_model = prune_model_weights(copy.deepcopy(original_model), target_layers=args.target_layers, pruning_percents=pruning_percents, algorithm=args.algorithm)
        method_key = 'pruned_weights'
    elif args.pruning_method == "prune_model_filters_by_importance":
        pruned_model = prune_model_filters_by_importance(copy.deepcopy(original_model), train_loader, test_loader, target_layers=args.target_layers, pruning_percents=pruning_percents,algorithm=args.algorithm)
        method_key = 'pruned_filters'
    else:
        raise ValueError("Invalid pruning method")

    # 프루닝된 모델 평가
    results[method_key] = evaluate_model_full(pruned_model, test_loader, device)

    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Target layers: {args.target_layers}")

    # 결과 출력
    print(f"Original Model: Accuracy: {results['original'][0]}, Inference Time: {results['original'][1]}, FLOPs: {results['original'][2]}, Non-zero Params: {results['original'][3]}")
    logging.info(f"Original Model: Accuracy: {results['original'][0]}, Inference Time: {results['original'][1]}, FLOPs: {results['original'][2]}, Non-zero Params: {results['original'][3]}")

    # 프루닝 결과 출력
    print(f"Pruned Model ({args.pruning_method}): Accuracy: {results[method_key][0]}, Inference Time: {results[method_key][1]}, FLOPs: {results[method_key][2]}, Non-zero Params: {results[method_key][3]}")
    logging.info(f"Pruned Model ({args.pruning_method}): Accuracy: {results[method_key][0]}, Inference Time: {results[method_key][1]}, FLOPs: {results[method_key][2]}, Non-zero Params: {results[method_key][3]}")

    return pruned_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pruning")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "ImageNet"], help="Dataset for pruning")
    parser.add_argument("--algorithm", type=str, default="kruskal", choices=["kruskal", "prim"], help="Algorithm for pruning")
    parser.add_argument("--pruning_method", type=str, default="prune_model_weights", choices=["prune_model_weights", "prune_model_filters_by_importance"], help="Pruning method")
    parser.add_argument("--target_layers", nargs="+",
                        default=['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight',
                                 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight',
                                 'layer2.0.conv2.weight', 'layer2.s1.conv1.weight', 'layer2.1.conv2.weight',
                                 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight',
                                 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight',
                                 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
                        help="Target layers for pruning")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()
    pruned_model = main(args)

    save_directory = 'saved_models/Resnet18'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    filename = f"{args.pruning_method}_algorithm_{args.algorithm}_model_{args.dataset}_weights.pth"
    # 프루닝된 모델의 가중치 저장
    save_path = os.path.join(save_directory, filename)
    torch.save(pruned_model.state_dict(), save_path)
    print(f"Pruned model weights saved to {save_path}")