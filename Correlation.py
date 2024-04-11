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
        filter_relations = []
        for i in range(height):
            for j in range(width):
                current_index = i * width + j
                if j < width - 1:
                    right_index = i * width + (j + 1)
                    filter_relations.append([abs(weights[k, 0, i, j] - weights[k, 0, i, j + 1]), str(current_index), str(right_index)])
                if i < height - 1:
                    down_index = (i + 1) * width + j
                    filter_relations.append([abs(weights[k, 0, i, j] - weights[k, 0, i + 1, j]), str(current_index), str(down_index)])
                if i < height - 1 and j < width - 1:
                    diag_index = (i + 1) * width + (j + 1)
                    filter_relations.append([abs(weights[k, 0, i, j] - weights[k, 0, i + 1, j + 1]), str(current_index), str(diag_index)])
        relations.append(filter_relations)

    return relations

conv1_weights = model.state_dict()['conv1.weight'].cpu().detach().numpy()
relations = generate_weight_relations(conv1_weights)

# Save relations to separate text files
for i, filter_relations in enumerate(relations):
    with open(f'relations/relations_conv1_{i}.txt', 'w') as f:
        for relation in filter_relations:
            f.write(str(relation) + '\n')