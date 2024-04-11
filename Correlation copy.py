import torch
import torch.nn as nn
import numpy as np

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
model.load_state_dict(torch.load('model_weights_70.pth'))

def generate_weight_relations(weights):
    relations = []
    num_filters, _, height, width = weights.shape

    for k in range(num_filters):
        for i in range(height):
            for j in range(width):
                current_index = i * width + j
                if j < width - 1:
                    right_index = i * width + (j + 1)
                    relations.append([abs(weights[k, 0, i, j] - weights[k, 0, i, j + 1]), str(current_index), str(right_index)])
                if i < height - 1:
                    down_index = (i + 1) * width + j
                    relations.append([abs(weights[k, 0, i, j] - weights[k, 0, i + 1, j]), str(current_index), str(down_index)])
                if i < height - 1 and j < width - 1:
                    diag_index = (i + 1) * width + (j + 1)
                    relations.append([abs(weights[k, 0, i, j] - weights[k, 0, i + 1, j + 1]), str(current_index), str(diag_index)])

    return relations



class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def kruskal(edges, n):
    edges.sort()  # 가중치 기준으로 간선 정렬
    uf = UnionFind(n)
    mst = []
    for edge in edges:
        weight, u, v = edge
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append(edge)
    return mst

def lightweight(weights):
    relations = generate_weight_relations(weights)
    print(relations)
    edges = [(relation[0], int(relation[1]), int(relation[2])) for relation in relations]  # 간선 형식으로 변환
    mst = kruskal(edges, len(weights[0].flatten()))  # 크루스칼 알고리즘 적용
    removed_indices = set(range(len(relations))) - set([edge[1] for edge in mst])  # 제거할 가중치 인덱스 추출
    lightweight_weights = weights.copy().flatten()
    lightweight_weights[list(removed_indices)] = 0  # 불필요한 가중치를 0으로 설정
    return lightweight_weights.reshape(weights.shape)  # 다시 형태 복원

# 경량화된 가중치 생성
conv1_weights = model.state_dict()['conv1.weight'].cpu().detach().numpy()
lightweight_conv1_weights = lightweight(conv1_weights)
print(lightweight_conv1_weights)
