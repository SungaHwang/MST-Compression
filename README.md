# Method
![MST-fig1](https://github.com/SungaHwang/MST-Compression/assets/74399508/96e216ab-9d3b-408c-8f8a-591a49cccefd)


## Pruning Algorithm

### Algorithm 1: Weight-Based MST Pruning
Below is the pseudocode for the weight-based MST pruning algorithm used in our model compression. The process leverages a minimum spanning tree (MST) to identify and remove the weakest connections within each layer of the network.

```plaintext
Inputs:
M - Original Model, L - List of layer names for pruning, P - Percentage of weights to be pruned in each layer, A - Algorithm for MST calculation
Output:
M - Compressed model

Procedure PruneWeights(M, L, P, A)
  Begin
    total_pruned <- 0
    for each (name, param) in M.named_parameters() do
      if name ∈ L then
        weights <- param.data.cpu().numpy()
        F, C, H, W <- shape(weights)
        for f <- 1 to F do
          G <- create graph from weights[f]
          T <- MST(G, A)
          prune_num <- (P / 100) * |nodes(G)|
          prune_set <- SelectWeakestNodes(T, prune_num)
          for node in prune_set do
            weights[f, node.c, node.i, node.j] <- 0
        param.data <- torch.from_numpy(weights)
        total_pruned <- total_pruned + 1
    return M
  End
```

### Algorithm 2: Filter Importance-Based MST Pruning
Below is the pseudocode for the filter Importance-Based MST pruning algorithm used in our model compression. This algorithm aims to prune filters in a model based on their importance using a minimum spanning tree (MST) approach to efficiently reduce model complexity while maintaining performance.

```plaintext
Inputs:
M - Original Model, L - List of layer names for pruning, P - Percentage of weights to be pruned in each layer, A - Algorithm for MST calculation
Output:
M - Compressed model

Procedure PruneFilters(M, L, P A)
Begin
    total_pruned ← 0
    for each (name, param) in M.named_parameters() do
        if name ∈ L then
            scores ← ComputeImportance(M, name)
            G ← create empty graph
            for i = 0 to length(scores)-1 do
                for j = i+1 to length(scores)-1 do
                    G.add_edge(i, j, weight=|scores[i] - scores[j]|)
            T ← MST(G, weight='weight', A)
            edges ← sort(T.edges(data=True), by weight, descending)
            prune_num ← ⌈(P / 100) * num⌉
            prune_set <- SelectWeakestNodes(T, prune_num)
            mask ← tensor of ones(num, dtype=float32)
            mask[prune_set] ← 0
            param.data *= mask.reshape(1, num, 1, 1)
            total_pruned += size(prune_set)
    return M
End
```


# Demo Page
https://github.com/SungaHwang/Model-Compression/assets/74399508/c6d1ebf0-bdd2-45d4-81d0-1c8f2024b274

