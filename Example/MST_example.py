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
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

def kruskal(edges, num_nodes):
    edges.sort()
    min_spanning_tree = []
    uf = UnionFind(num_nodes)

    for edge in edges:
        weight, start, end = edge
        if uf.union(int(start), int(end)):
            min_spanning_tree.append(edge)
            if len(min_spanning_tree) == num_nodes - 1:
                break

    return min_spanning_tree

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

if __name__ == "__main__":

    matrix = [
        [0.04500538, '0', '1'],
        [0.15314113, '0', '3'],
        [0.14587583, '0', '4'],
        [0.4263518, '1', '2'],
        [0.19088121, '1', '4'],
        [0.14641781, '1', '5'],
        [0.5727696, '2', '5'],
        [0.0072652996, '3', '4'],
        [0.844815, '3', '6'],
        [0.98176974, '3', '7'],
        [0.33729902, '4', '5'],
        [0.9745045, '4', '7'],
        [0.59138197, '4', '8'],
        [0.25408295, '5', '8'],
        [0.13695472, '6', '7'],
        [0.38312247, '7', '8']
    ]

    edges = [[float(weight), start, end] for weight, start, end in matrix]

    min_spanning_tree = kruskal(edges, num_nodes=9)
    bottom_nodes, _ = get_bottom_nodes(min_spanning_tree)

    print("After Kruskal MST:")
    for edge in min_spanning_tree:
        print(edge)
    
    print("\n맨 아래에 있는 모든 노드:")
    for node in bottom_nodes:
        print("노드:", node)
    
    weights = remove_bottom_nodes(bottom_nodes, [0.04500538, 0.15314113, 0.14587583, 0.4263518, 0.19088121, 0.14641781, 0.5727696, 0.0072652996, 0.844815])
    print("\n맨 아래에 있는 모든 노드 가중치 제거 결과:")
    print(weights)

