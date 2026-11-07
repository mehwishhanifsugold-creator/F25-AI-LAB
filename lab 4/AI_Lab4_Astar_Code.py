import heapq

def astar(graph, h, start, goal):
    pq = [(h[start], 0, start, [start])]
    best = {start: 0}
    expanded = []

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if g != best[node]:
            continue

        if node == goal:
            return g, path, expanded

        expanded.append(node)

        for nxt, w in graph.get(node, []):
            ng = g + w

            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                heapq.heappush(
                    pq, (ng + h[nxt], ng, nxt, path + [nxt])
                )

    return None, [], expanded


def ucs(graph, start, goal):
    pq = [(0, start, [start])]
    best = {start: 0}
    expanded = []

    while pq:
        g, node, path = heapq.heappop(pq)

        if g != best[node]:
            continue

        if node == goal:
            return g, path, expanded

        expanded.append(node)

        for nxt, w in graph.get(node, []):
            ng = g + w

            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                heapq.heappush(pq, (ng, nxt, path + [nxt]))

    return None, [], expanded


graph = {
    "S": [("A", 1), ("B", 4)],
    "A": [("C", 2)],
    "B": [("G", 5)],
    "C": [("G", 3)],
    "G": []
}

h = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}

# Task 1 and Task 2: A* with the supplied heuristic
a_cost, a_path, a_expanded = astar(graph, h, "S", "G")
u_cost, u_path, u_expanded = ucs(graph, "S", "G")

print("TASK 1: A* SEARCH")
print("Path:", " -> ".join(a_path))
print("Cost:", a_cost)
print("Expanded nodes:", a_expanded)
print("Expansion count:", len(a_expanded))

print("\nTASK 2: A* AND UCS COMPARISON")
print("A* path:", " -> ".join(a_path))
print("A* cost:", a_cost)
print("A* expanded nodes:", a_expanded)
print("A* expansion count:", len(a_expanded))
print("UCS path:", " -> ".join(u_path))
print("UCS cost:", u_cost)
print("UCS expanded nodes:", u_expanded)
print("UCS expansion count:", len(u_expanded))

# Task 3: Change only h(C) from 2 to 10
h_over = {"S": 5, "A": 4, "B": 4, "C": 10, "G": 0}
over_cost, over_path, over_expanded = astar(
    graph, h_over, "S", "G"
)

print("\nTASK 3: OVERestimating HEURISTIC")
print("Changed h(C):", h_over["C"])
print("Path:", " -> ".join(over_path))
print("Cost:", over_cost)
print("Expanded nodes:", over_expanded)
print("Expansion count:", len(over_expanded))
