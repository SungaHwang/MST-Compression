import numpy as np

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False

def kruskal(matrix):
    edges = []
    n = len(matrix)
    
    for i in range(n):
        for j in range(i+1, n):
            edges.append((matrix[i][j], i, j))
    
    edges.sort()
    
    uf = UnionFind(n)
    mst = []
    mst_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            mst_weight += weight
    
    return mst, mst_weight

matrix = [
    [0, 0.13883805, 0.2665918, 0.31708145, -0.6050197],
    [0.14868219, 0, -0.46095154, 0.00719108, 0.30362144],
    [-0.29883927, 0.30510303, 0, 0.15816714, -0.19512102],
    [0.17021039, -0.37263504, -0.04782006, 0, -1.0081314],
    [0.42745575, 0.10338372, 0.35008624, 0.21098094, 0]
]

mst, total_weight = kruskal(matrix)
print("Minimum Spanning Tree:", mst)
print("Total Weight:", total_weight)

def compress_matrix(matrix, mst):
    n = len(matrix)
    compressed_matrix = np.zeros_like(matrix)
    
    for u, v, weight in mst:
        compressed_matrix[u][v] = weight
        compressed_matrix[v][u] = weight

    return compressed_matrix

compressed_matrix = compress_matrix(matrix, mst)
print("\nCompressed Weight Matrix (kruskal):")
print(compressed_matrix)
