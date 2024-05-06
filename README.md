# Method
![MST-fig1](https://github.com/SungaHwang/MST-Compression/assets/74399508/96e216ab-9d3b-408c-8f8a-591a49cccefd)

## Pruning Algorithm
### Algorithm 1: Weight-Based MST Pruning
Below is the pseudocode for the weight-based MST pruning algorithm used in our model compression. The process leverages a minimum spanning tree (MST) to identify and remove the weakest connections within each layer of the network.

```plaintext
Inputs: M - Original Model, L - List of layer names for pruning, P - Percentage of weights to be pruned in each layer, A - Algorithm for MST calculation
Output: M - Compressed model

Procedure PruneWeights(M, L, P, A)
  Begin
    total_pruned ← 0
    for each (name, param) in M.named_parameters() do
      if name ∈ L then
        weights ← param.data.cpu().numpy()
        F, C, H, W ← shape(weights)
        layer_pruned ← 0
        for f = 1 to F do
          G ← create empty graph
          for c = 1 to C do
            for i = 1 to H do
              for j = 1 to W do
                G.add_node((c, i, j))
                for (di, dj) in [(0, 1), (1, 0), (1, 1)]:
                  ni, nj = i + di, j + dj
                  if ni < H and nj < W then
                    G.add_edge((c, i, j), (c, ni, nj), weight=|weights[f, c, i, j] - weights[f, c, ni, nj]|)
          T ← MST(G, A)
          prune_num ← ⌈(P / 100) * |nodes(G)|⌉
          prune_set ← SelectWeakestNodes(T, prune_num)   
          for node in prune_set do
            c, i, j ← node
            weights[f, c, i, j] ← 0
            layer_pruned ← layer_pruned + 1
        total_pruned ← total_pruned + layer_pruned
        param.data ← torch.from_numpy(weights).to(param.device)
    return M
  End
```

### Algorithm 2: Filter Importance-Based MST Pruning
Below is the pseudocode for the filter Importance-Based MST pruning algorithm used in our model compression. This algorithm aims to prune filters in a model based on their importance using a minimum spanning tree (MST) approach to efficiently reduce model complexity while maintaining performance.

```plaintext
Inputs: M - Original Model, L - List of layer names for pruning, P - Percentage of filters to prune, A - Algorithm for MST calculation
Output: M - Compressed model

Procedure PruneFilters(M, L, P, A)
  Begin
    total_pruned ← 0
    for each (name, param) in M.named_parameters() do
      if name ∈ L then
        scores ← ComputeImportance(M, name)
        num ← length(scores)
        G ← create empty graph
        for i = 0 to num-1 do
          for j = i+1 to num-1 do
            G.add_edge(i, j, weight=-|scores[i] - scores[j]|)
        mst ← MST(G, weight='weight', A)
        edges ← sort(mst.edges(data=True), by weight, descending)
        prune_num ← ⌈(P / 100) * num⌉
        prune_set ← empty set
        for edge in edges (reverse order) do
          if size(prune_set) < prune_num then
            prune_set.add(edge[0], edge[1])
        mask ← tensor of ones (num, dtype=float32)
        mask[prune_set] ← 0
        param.data *= mask.reshape(1, num, 1, 1)
        total_pruned ← total_pruned + size(prune_set)
    return M
  End
```

