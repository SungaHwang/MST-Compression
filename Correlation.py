import numpy as np

weights = np.random.rand(3, 3) # 임의로

def generate_weight_relations(weights):
    relations = []
    n, m = weights.shape
    flattened_weights = weights.flatten()

    for i in range(n):
        for j in range(m):
            current_index = i * m + j
            if j < m - 1: 
                right_index = i * m + (j + 1)
                relations.append([flattened_weights[current_index] - flattened_weights[right_index], str(current_index), str(right_index)])
            if i < n - 1: 
                down_index = (i + 1) * m + j
                relations.append([flattened_weights[current_index] - flattened_weights[down_index], str(current_index), str(down_index)])
            if i < n - 1 and j < m - 1:
                diag_index = (i + 1) * m + (j + 1)
                relations.append([flattened_weights[current_index] - flattened_weights[diag_index], str(current_index), str(diag_index)])

    relations = sorted(relations, key=lambda x: np.abs(x[0]))

    for relation in relations:
        relation[0] = np.abs(relation[0])

    return relations

weight_relations = generate_weight_relations(weights)

for relation in weight_relations:
    print(relation)
