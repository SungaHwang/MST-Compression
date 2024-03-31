import numpy as np

def floyd_warshall(matrix):
    n = len(matrix)
    dist = np.array(matrix)
    next_node = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j or matrix[i][j] == 0:
                dist[i][j] = float('inf')
            next_node[i][j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node

def reconstruct_path(u, v, next_node):
    if next_node[u][v] == 0:
        return []
    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)
    return path

def compress_matrix(matrix, next_node):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] != 0:
                path = reconstruct_path(i, j, next_node)
                if len(path) > 2 or (len(path) == 2 and path[1] != j):
                    matrix[i][j] = 0
    return matrix

matrix = [
    [0, 0.13883805, 0.2665918, 0.31708145, -0.6050197],
    [0.14868219, 0, -0.46095154, 0.00719108, 0.30362144],
    [-0.29883927, 0.30510303, 0, 0.15816714, -0.19512102],
    [0.17021039, -0.37263504, -0.04782006, 0, -1.0081314],
    [0.42745575, 0.10338372, 0.35008624, 0.21098094, 0]
]

dist, next_node = floyd_warshall(matrix)
compressed_matrix = compress_matrix(matrix, next_node)

print("\nCompressed Weight Matrix (floyd):")
print(compressed_matrix)