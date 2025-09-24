import heapq
import matplotlib.pyplot as plt
import networkx as nx

# GRAPH DATA
graph = {
    "Cilegon": {"Tangerang": 81},
    "Tangerang": {"Cilegon": 81, "Jakarta": 29},
    "Jakarta": {"Tangerang": 29, "Bekasi": 22, "Depok": 25},
    "Bekasi": {"Depok": 41, "Jakarta": 22, "Subang": 95, "Indramayu": 185},
    "Depok": {"Jakarta": 25, "Bekasi": 41, "Bogor": 44},
    "Bogor": {"Depok": 44, "Sukabumi": 57},
    "Sukabumi": {"Bogor": 57, "Bandung": 93},
    "Bandung": {"Sukabumi": 93, "Cirebon": 106, "Tasikmalaya": 83},
    "Subang": {"Bekasi": 95, "Cirebon": 103},
    "Indramayu": {"Bekasi": 185, "Cirebon": 56},
    "Cirebon": {"Subang": 103, "Bandung": 106, "Indramayu": 56, "Tegal": 71},
    "Tegal": {"Cirebon": 71, "Purwokerto": 65, "Pekalongan": 70},
    "Pekalongan": {"Tegal": 70, "Semarang": 83},
    "Semarang": {"Pekalongan": 83, "Kudus": 60, "Ambarawa": 37},
    "Ambarawa": {"Magelang": 35, "Semarang": 37, "Surakarta": 70},
    "Magelang": {"Purwokerto": 107, "Ambarawa": 37, "Yogyakarta": 40},
    "Yogyakarta": {"Kebumen": 81, "Magelang": 40, "Pacitan": 107, "Surakarta": 75},
    "Cilacap": {"Tasikmalaya": 96, "Purwokerto": 42, "Kebumen": 70},
    "Tasikmalaya": {"Bandung": 83, "Purwokerto": 113, "Cilacap": 96},
    "Purwokerto": {"Tasikmalaya": 113, "Tegal": 65, "Kebumen": 51, "Cilacap": 42, "Magelang": 107},
    "Kebumen": {"Purwokerto": 51, "Cilacap": 70, "Yogyakarta": 81},
    "Surakarta": {"Ngawi": 83, "Ambarawa": 70, "Yogyakarta": 75},
    "Ngawi": {"Surakarta": 83, "Bojonegoro": 72, "Nganjuk": 73},
    "Rembang": {"Kudus": 62, "Bojonegoro": 103, "Tuban": 93},
    "Kudus": {"Semarang": 60, "Rembang": 62},
    "Nganjuk": {"Ngawi": 73, "Trenggalek": 86, "Sidoarjo": 118},
    "Trenggalek": {"Pacitan": 106, "Nganjuk": 86, "Kepanjen": 114},
    "Pacitan": {"Yogyakarta": 107, "Trenggalek": 106},
    "Bojonegoro": {"Surabaya": 111, "Rembang": 103, "Ngawi": 72},
    "Tuban": {"Rembang": 93, "Surabaya": 95},
    "Surabaya": {"Tuban": 95, "Bojonegoro": 111, "Sidoarjo": 35},
    "Sidoarjo": {"Surabaya": 35, "Nganjuk": 118, "Kepanjen": 108, "Probolinggo": 66},
    "Probolinggo": {"Sidoarjo": 66, "Lumajang": 75, "Situbundo": 100},
    "Situbundo": {"Probolinggo": 100, "Banyuwangi": 88},
    "Lumajang": {"Kepanjen": 116, "Probolinggo": 75, "Jember": 65},
    "Kepanjen": {"Trenggalek": 114, "Sidoarjo": 108, "Lumajang": 116},
    "Jember": {"Lumajang": 65, "Banyuwangi": 100},
    "Banyuwangi": {"Situbundo": 88, "Jember": 100}
}

# NILAI HEURISTIC
heuristic = {
    "Cilegon": 950, "Tangerang": 882, "Jakarta": 861, "Bekasi": 840,
    "Depok": 858, "Bogor": 852, "Sukabumi": 832, "Bandung": 780,
    "Subang": 750, "Indramayu": 700, "Cirebon": 662, "Tasikmalaya": 684,
    "Tegal": 595, "Purwokerto": 571, "Cilacap": 592, "Pekalongan": 537,
    "Kebumen": 521, "Magelang": 463, "Semarang": 455, "Ambarawa": 445,
    "Yogyakarta": 440, "Kudus": 416, "Surakarta": 396, "Rembang": 373,
    "Ngawi": 334, "Pacitan": 357, "Bojonegoro": 298, "Tuban": 295,
    "Nganjuk": 276, "Trenggalek": 289, "Surabaya": 210, "Sidoarjo": 196,
    "Kepanjen": 197, "Probolinggo": 132, "Lumajang": 122, "Situbundo": 70,
    "Jember": 71, "Banyuwangi": 0
}
# HITUNG TOTAL BIAYA JALUR / TOTAL COST OF PATH
def total_cost(path):
    cost = 0
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]
    return cost

# GREEDY BEST FIRST SEARCH
def gbfs(start, goal):
    queue = [(heuristic[start], [start])]
    visited = set()
    while queue:
        h, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            for neighbor in graph[node]:
                heapq.heappush(queue, (heuristic[neighbor], path+[neighbor]))
            visited.add(node)
    return None

# UNIFORM COST SEARCH
def ucs(start, goal):
    queue = [(0, [start])]
    visited = set()
    while queue:
        cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return path, cost
        if node not in visited:
            for neighbor, distance in graph[node].items():
                heapq.heappush(queue, (cost+distance, path+[neighbor]))
            visited.add(node)
    return None, float("inf")

# ITERATIVE DEEPENING SEARCH
def dls(node, goal, limit, path, visited):
    if node == goal:
        return path
    if limit <= 0:
        return None
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            result = dls(neighbor, goal, limit-1, path+[neighbor], visited)
            if result:
                return result
    return None

def ids(start, goal, max_depth=50):
    for depth in range(max_depth):
        visited = set()
        result = dls(start, goal, depth, [start], visited)
        if result:
            return result
    return None

# A STAR SEARCH
def astar(start, goal):
    queue = [(heuristic[start], 0, [start])]
    visited = set()
    while queue:
        f, g, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return path, g
        if node not in visited:
            for neighbor, distance in graph[node].items():
                new_g = g + distance
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(queue, (new_f, new_g, path+[neighbor]))
            visited.add(node)
    return None, float("inf")

# VISUALISASI GRAPH
def draw_graph(path=None, title=""):
    G = nx.Graph()
    for city, neighbors in graph.items():
        for neighbor, dist in neighbors.items():
            G.add_edge(city, neighbor, weight=dist)

    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Semua node & edge
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6)

    # Highlight path
    if path:
        edges_in_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path,
                               edge_color="red", width=2)
        nx.draw_networkx_nodes(G, pos, nodelist=path,
                               node_color="yellow", node_size=600)

    plt.title(title)
    plt.axis("off")
    plt.show()

# MAIN PROGRAM
if __name__ == "__main__":
    start = "Cilegon"
    goal = "Banyuwangi"

    print("<---> Greedy Best First Search <--->")
    gbfs_path = gbfs(start, goal)
    print("Jalur :", " → ".join(gbfs_path))
    print("Total Jarak :", total_cost(gbfs_path), "km")

    print("\n<---> Uniform Cost Search <--->")
    ucs_path, ucs_cost = ucs(start, goal)
    print("Jalur :", " → ".join(ucs_path))
    print("Total Jarak :", ucs_cost, "km")

    print("\n<---> Iterative Deepening Search <--->")
    ids_path = ids(start, goal)
    print("Jalur :", " → ".join(ids_path))
    print("Total Jarak :", total_cost(ids_path), "km")

    print("\n<---> A* Search <--->")
    astar_path, astar_cost = astar(start, goal)
    print("Jalur :", " → ".join(astar_path))
    print("Total Jarak :", astar_cost, "km")
    
    # Visualisasi hasil pencarian
    draw_graph(gbfs_path, "Greedy Best First Search")
    draw_graph(ucs_path, "Uniform Cost Search")
    draw_graph(ids_path, "Iterative Deepening Search")
    draw_graph(astar_path, "A* Search")    