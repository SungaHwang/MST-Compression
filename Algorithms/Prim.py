class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, edge):
        self.heap.append(edge)
        self.heapify_up(len(self.heap) - 1)

    def extract_min(self):
        if self.heap:
            min_edge = self.heap[0]
            last_edge = self.heap.pop()
            if self.heap:
                self.heap[0] = last_edge
                self.heapify_down(0)
            return min_edge
        return None

    def heapify_up(self, index):
        while index > 0:
            parent_index = (index - 1) // 2
            if self.heap[parent_index][0] > self.heap[index][0]:
                self.heap[parent_index], self.heap[index] = self.heap[index], self.heap[parent_index]
                index = parent_index
            else:
                break

    def heapify_down(self, index):
        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest = index
            if left_child_index < len(self.heap) and self.heap[left_child_index][0] < self.heap[smallest][0]:
                smallest = left_child_index
            if right_child_index < len(self.heap) and self.heap[right_child_index][0] < self.heap[smallest][0]:
                smallest = right_child_index
            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

def prim(edges, num_nodes):
    graph = {}
    for weight, u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append((v, weight))
        graph[v].append((u, weight))

    visited = set()
    min_heap = MinHeap()
    min_spanning_tree = []
    start_vertex = list(graph.keys())[0]
    visited.add(start_vertex)
    for neighbor, weight in graph[start_vertex]:
        min_heap.insert((weight, start_vertex, neighbor))
    while min_heap.heap:
        weight, u, v = min_heap.extract_min()
        if v not in visited:
            visited.add(v)
            min_spanning_tree.append((weight, u, v))
            for neighbor, weight in graph[v]:
                if neighbor not in visited:
                    min_heap.insert((weight, v, neighbor))
    return min_spanning_tree