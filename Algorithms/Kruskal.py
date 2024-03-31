def find(x):
    if uf[x] != x:
        uf[x] = find(uf[x])
    return uf[x]

def union(x, y):
    x, y = find(x), find(y)
    if x != y:
        uf[min(x, y)] = max(x, y)

def kruskal(graph):

    nodes = list(graph.keys())
    edges = []
    for node in graph:
        for neighbor, weight in graph[node].items():
            edges.append((node, neighbor, weight))

    edges.sort(key=lambda x: x[2])

    mst = []
    count = 0
    cost = 0

    for edge in edges:
        u, v, w = edge
        if find(u) != find(v):
            union(u, v)
            mst.append(edge)
            cost += w
            count += 1

    if count == len(nodes) - 1:
        return mst, cost
    return False

if __name__ == "__main__":
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }

    uf = {node: node for node in graph.keys()}

    result = kruskal(graph)
    print(result)