import heapq
import numpy as np

def prim_algorithm(matrix):
    n = len(matrix)
    m = len(matrix[0])
    visited = [False] * m
    min_heap = []
    min_spanning_tree = []

    start_vertex = 0
    visited[start_vertex] = True
    for j in range(m):
        if matrix[start_vertex][j] != 0:
            heapq.heappush(min_heap, (matrix[start_vertex][j], start_vertex, j))

    while min_heap:
        weight, u, v = heapq.heappop(min_heap)
        if not visited[v]:
            visited[v] = True
            min_spanning_tree.append((u, v, weight))
            for j in range(m):
                if matrix[v][j] != 0 and not visited[j]:
                    heapq.heappush(min_heap, (matrix[v][j], v, j))

    return min_spanning_tree

matrix = [
    [0.13883805,  0.2665918,   0.31708145, -0.6050197],
    [0.14868219, -0.46095154,  0.00719108,  0.30362144],
    [-0.29883927,  0.30510303, 0.15816714, -0.19512102],
    [0.17021039, -0.37263504, -0.04782006, -1.0081314],
    [0.42745575,  0.10338372,  0.35008624,  0.21098094],
    [-0.21147376, -0.13646333,  0.16905327, -0.12487327],
    [0.47969162, -1.12304, -0.27964967, 0.15170094],
    [0.04808323,  0.26588908, -0.62521964,  0.08806055]
]

min_spanning_tree = prim_algorithm(matrix)
print(min_spanning_tree)

for edge in min_spanning_tree:
    print(f"{edge[0]} - {edge[1]} : {edge[2]}")


def compress_matrix(matrix, min_spanning_tree):
    n = len(matrix)
    compressed_matrix = np.zeros_like(matrix)

    for edge in min_spanning_tree:
        u, v, weight = edge
        compressed_matrix[u][v] = weight
        compressed_matrix[v][u] = weight

    return compressed_matrix

compressed_matrix = compress_matrix(matrix, min_spanning_tree)
print("\nCompressed Weight Matrix (prim):")
print(compressed_matrix)