from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import numpy as np
import networkx as nx
import copy
from flask_socketio import SocketIO, emit
import markdown
import os
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

pruning_percents = {
    'conv1.weight': 0,
    'layer1.0.conv1.weight': 0,
    'layer1.0.conv2.weight': 0,
    'layer1.1.conv1.weight': 5,
    'layer1.1.conv2.weight': 5,
    'layer2.0.conv1.weight': 0,
    'layer2.0.conv2.weight': 0,
    'layer2.1.conv1.weight': 0,
    'layer2.1.conv2.weight': 5,
    'layer3.0.conv1.weight': 5,
    'layer3.0.conv2.weight': 5,
    'layer3.1.conv1.weight': 40,
    'layer3.1.conv2.weight': 30,
    'layer4.0.conv1.weight': 30,
    'layer4.0.conv2.weight': 40,
    'layer4.1.conv1.weight': 70,
    'layer4.1.conv2.weight': 70
}

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
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader

def initialize_model():
    model = resnet18(pretrained=False)
    model.to(device)
    return model

def train_model(model, train_loader, epochs=100):
    model = model.to(device)
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
        progress = (epoch + 1) / epochs * 100
        socketio.emit('log', {'message': f'Epoch {epoch+1}/{epochs} completed', 'progress': progress})
    return model

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

def prune_model_weights(model, target_layers, pruning_percents, algorithm='kruskal'):
    model = model.to(device)
    total_pruned_weights = 0
    for name, param in model.named_parameters():
        if name in target_layers:
            pruning_percent = pruning_percents[name]
            weights = param.data.cpu().numpy()
            F, C, H, W = weights.shape

            for f in range(F):
                G = nx.Graph()
                for c in range(C):
                    for i in range(H):
                        for j in range(W):
                            G.add_node((c, i, j))
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

                T = nx.minimum_spanning_tree(G, algorithm=algorithm)
                num_nodes_to_prune = int(G.number_of_nodes() * (pruning_percent / 100))
                nodes_to_prune = set()

                for edge in reversed(list(T.edges(data=True))):
                    if len(nodes_to_prune) < num_nodes_to_prune:
                        nodes_to_prune.add(edge[0])
                    else:
                        break

                for node in nodes_to_prune:
                    c, i, j = node
                    weights[f, c, i, j] = 0

            param.data = torch.from_numpy(weights).to(param.device)
            socketio.emit('log', {'message': f'Pruned {name} by {pruning_percent}%', 'progress': 100})

    return model

def prune_model_filters_by_importance(model, train_loader, test_loader, target_layers, pruning_percents, algorithm='prim'):
    model = model.to(device)
    total_pruned_filters = 0
    for name, param in model.named_parameters():
        if name in target_layers:
            importance_scores = compute_layer_importance(model, train_loader, test_loader, name)
            num_filters = len(importance_scores)

            G = nx.Graph()
            for i in range(num_filters):
                for j in range(i + 1, num_filters):
                    G.add_edge(i, j, weight=-np.abs(importance_scores[i] - importance_scores[j]))

            mst = nx.minimum_spanning_tree(G, weight='weight', algorithm=algorithm)
            sorted_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

            pruning_percent = pruning_percents[name]
            num_filters_to_prune = int(num_filters * pruning_percent / 100)
            pruned_filters = set()

            for edge in sorted_edges[::-1]:
                if len(pruned_filters) < num_filters_to_prune:
                    pruned_filters.add(edge[1])
                if len(pruned_filters) >= num_filters_to_prune:
                    break

            pruned_filters = list(pruned_filters)
            mask = torch.ones(num_filters, dtype=torch.float32, device=device)
            mask[pruned_filters] = 0
            param.data *= mask[:, None, None, None]

            socketio.emit('log', {'message': f'Pruned {name} by {pruning_percent}%', 'progress': 100})
            total_pruned_filters += len(pruned_filters)

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

def compute_flops(model, input_size=(1, 3, 224, 224)):
    model = model.to(device)
    model.eval()
    input = torch.randn(input_size).to(device)

    total_flops = 0

    def conv_flops_hook(module, input, output):
        weight = module.weight
        active_elements_count = torch.sum(weight != 0).item()
        output_dims = output.shape[2:]
        kernel_dims = module.kernel_size
        in_channels = module.in_channels / module.groups
        flops = torch.prod(torch.tensor(kernel_dims)).item() * in_channels * active_elements_count * torch.prod(torch.tensor(output_dims)).item()

        nonlocal total_flops
        total_flops += flops

    hooks = []

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(conv_flops_hook)
            hooks.append(hook)

    with torch.no_grad():
        model(input)

    for hook in hooks:
        hook.remove()

    return total_flops

def count_nonzero_parameters(model):
    model = model.to(device)
    total_params = 0
    zero_count = 0
    for param in model.parameters():
        param_count = param.numel()
        zero_count += torch.sum(param == 0).item()
        total_params += param_count

    nonzero_count = total_params - zero_count
    return nonzero_count

def evaluate_model_full(model, test_loader):
    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    correct = 0
    total = 0
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

    flops = compute_flops(model)
    nonzero_params = count_nonzero_parameters(model)

    return accuracy, inference_time, flops, nonzero_params

def fine_tune_pruned_model(pruned_model, train_loader, test_loader, epochs=10):
    optimizer = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    pruned_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = pruned_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in pruned_model.named_parameters():
                if 'weight' in name:
                    zero_mask = (param == 0)
                    if zero_mask.any():
                        param.grad.data[zero_mask] = 0

            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                socketio.emit('log', {'message': f'Fine-tuning epoch {epoch+1}, batch {i+1}, loss: {running_loss / 100}'})
                running_loss = 0.0

    return pruned_model

@app.route('/')
def index():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        readme_html = markdown.markdown(content, extensions=['fenced_code', 'codehilite'])
    
    return render_template('index.html', readme_html=readme_html)

@app.route('/prune', methods=['POST'])
def prune():
    data = request.get_json()
    dataset = data['dataset']
    algorithm = data['algorithm']
    pruning_method = data['pruning_method']
    epochs = data['epochs']

    train_loader, test_loader = load_data(dataset)
    model = initialize_model()

    save_directory = 'saved_models/Resnet18'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    filename = f"original_model_{dataset}.pth"
    save_path = os.path.join(save_directory, filename)

    if not os.path.exists(save_path):
        trained_model = train_model(model, train_loader, epochs=epochs)
        torch.save(trained_model.state_dict(), save_path)
    else:
        trained_model = initialize_model()
        trained_model.load_state_dict(torch.load(save_path))
        trained_model.eval()

    original_model = copy.deepcopy(trained_model)
    results = {}
    
    results['initial'] = evaluate_model_full(original_model, test_loader)
    if pruning_method == "prune_model_weights":
        pruned_model = prune_model_weights(copy.deepcopy(original_model), target_layers=data['target_layers'], pruning_percents=pruning_percents, algorithm=algorithm)
        method_key = 'pruned_weights'
    elif pruning_method == "prune_model_filters_by_importance":
        pruned_model = prune_model_filters_by_importance(copy.deepcopy(original_model), train_loader, test_loader, target_layers=data['target_layers'], pruning_percents=pruning_percents,algorithm=algorithm)
        method_key = 'pruned_filters'
    else:
        return jsonify({'error': 'Invalid pruning method'}), 400

    results[method_key] = evaluate_model_full(pruned_model, test_loader)
    pruned_model = fine_tune_pruned_model(pruned_model, train_loader, test_loader)
    fine_tuned_results = evaluate_model_full(pruned_model, test_loader)
    results[method_key + '_fine_tuned'] = fine_tuned_results
    
    return jsonify({
        'initial_accuracy': results['initial'][0],
        'trained_accuracy': results['initial'][0],
        'pruned_accuracy': results[method_key][0],
        'fine_tuned_accuracy': fine_tuned_results[0],
        'initial_flops': results['initial'][2],
        'trained_flops': results['initial'][2],
        'pruned_flops': results[method_key][2],
        'fine_tuned_flops': fine_tuned_results[2],
        'initial_params': results['initial'][3],
        'trained_params': results['initial'][3],
        'pruned_params': results[method_key][3],
        'fine_tuned_params': fine_tuned_results[3]
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    socketio.run(app, debug=True)
