def floyd_warshall(edges, num_nodes):
   
    INF = float('inf')
    
    distance = [[INF] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        distance[i][i] = 0

    for weight, u, v in edges:
        distance[u][v] = min(weight, distance[u][v])
        distance[v][u] = min(weight, distance[v][u])

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]

    return distance
