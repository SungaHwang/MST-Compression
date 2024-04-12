class DisjointSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root1 = self.find(u)
        root2 = self.find(v)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1


def kruskal(edges, num_nodes):
    edges.sort() 
    min_spanning_tree = []
    disjoint_set = DisjointSet(num_nodes)

    for edge in edges:
        weight, u, v = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            min_spanning_tree.append(edge)

    return min_spanning_tree