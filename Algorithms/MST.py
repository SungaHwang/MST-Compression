import torch
import torch.nn as nn
import numpy as np
from file_reader import read_data_from_file
from Kruskal import kruskal
from Prim import prim
from Floyd import floyd

def get_bottom_nodes(min_spanning_tree):
    bottom_nodes = set()
    all_nodes = set()
    for edge in min_spanning_tree:
        _, start, end = edge
        all_nodes.add(start)
        all_nodes.add(end)
        if start not in [x[1] for x in min_spanning_tree]:
            bottom_nodes.add(start)
        elif end not in [x[1] for x in min_spanning_tree]:
            bottom_nodes.add(end)
    return bottom_nodes, all_nodes

def remove_bottom_nodes(bottom_nodes, weights):
    for node in bottom_nodes:
        weights[int(node)] = 0
    return weights

def process_data_for_filters(num_filters):
    all_bottom_nodes_per_filter = {}

    for i in range(num_filters):
        filter_name = str(i)
        filter_path = f"relations/relations_conv1_{i}.txt"

        with open(filter_path, 'r') as file:
            matrix = []
            for line in file:
                line = line.strip().replace('[', '').replace(']', '').replace("'", '')
                parts = line.strip().split(',')
                weight = float(parts[0].strip())
                start = int(parts[1].strip())
                end = int(parts[2].strip())
                matrix.append([weight, start, end])

        edges = [[weight, start, end] for weight, start, end in matrix]

        min_spanning_tree = kruskal(edges, num_nodes=9) # 사용할 알고리즘으로 변경
        bottom_nodes, _ = get_bottom_nodes(min_spanning_tree)

        all_bottom_nodes_per_filter[filter_name] = bottom_nodes

    return all_bottom_nodes_per_filter

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
    
model = CNNModel()
weights = model.state_dict()['conv1.weight'][0].cpu().detach().numpy()

if __name__ == "__main__":
    num_filters = 16 

    all_bottom_nodes_per_filter = process_data_for_filters(num_filters)
    all_weights_per_filter = []

    for filter_name, bottom_nodes in all_bottom_nodes_per_filter.items():
        print(f"Filter {filter_name}:")
        print("Bottom Nodes:")
        for node in bottom_nodes:
            print(node)
        model_path = 'model_weights_70.pth'
        model_weight = torch.load(model_path)['conv1.weight'][int(filter_name)].tolist()
        model_weight = model_weight[0]
        model_weights = model_weight[0] + model_weight[1] + model_weight[2]

        weights = remove_bottom_nodes(bottom_nodes, model_weights)
        print("\n맨 아래에 있는 모든 노드 가중치 제거 결과:")
        print(weights)
        conv1_weights_tensor = torch.Tensor(weights).view(1,3,3)
        print(conv1_weights_tensor)
        all_weights_per_filter.append(weights)
    
    torch.set_printoptions(precision=10)
    #weights_tensors = [torch.tensor(weight, dtype=torch.float32).view(3, 3) for weight in all_weights_per_filter]
    #print(weights_tensors)

def load_weights(model, weights):
    for i, weight in enumerate(weights):
        conv_weight = torch.tensor(weight, dtype=torch.float32).view(1, 1, 3, 3)
        model.conv1.weight.data[i] = conv_weight

load_weights(model, all_weights_per_filter)
print(model.conv1.weight)

# 모델 저장
import os
os.makedirs('new_model', exist_ok = True)
torch.save(model.state_dict(), 'new_model/kruskal_weights.pth')