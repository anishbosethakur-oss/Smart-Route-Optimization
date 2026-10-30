import networkx as nx
import matplotlib.pyplot as plt
import random
import csv

# ============================================================
# SMART ROUTE OPTIMIZATION SYSTEM
# Member 1 - Network Simulation + Dijkstra + Metrics
# ============================================================

# Create network
network = nx.Graph()

# ============================================================
# 1. ADD ROUTERS
# ============================================================

routers = ["R1", "R2", "R3", "R4", "R5", "R6"]
network.add_nodes_from(routers)

# ============================================================
# 2. ADD LINKS
# ============================================================

network.add_edge("R1", "R2", delay=10, bandwidth=100)
network.add_edge("R1", "R3", delay=15, bandwidth=100)
network.add_edge("R2", "R4", delay=10, bandwidth=80)
network.add_edge("R3", "R4", delay=20, bandwidth=100)
network.add_edge("R3", "R5", delay=10, bandwidth=50)
network.add_edge("R4", "R5", delay=5, bandwidth=100)
network.add_edge("R2", "R6", delay=12, bandwidth=70)
network.add_edge("R6", "R5", delay=8, bandwidth=80)

# ============================================================
# 3. GENERATE DYNAMIC UTILIZATION
# ============================================================

for source, destination in network.edges():

    utilization = random.uniform(0.20, 0.90)

    network[source][destination]["utilization"] = utilization

# ============================================================
# 4. CALCULATE TRAFFIC
# ============================================================

for source, destination in network.edges():

    bandwidth = network[source][destination]["bandwidth"]
    utilization = network[source][destination]["utilization"]

    traffic = bandwidth * utilization

    network[source][destination]["traffic"] = traffic

# ============================================================
# 5. DETECT CONGESTION
# ============================================================

for source, destination in network.edges():

    utilization = network[source][destination]["utilization"]

    if utilization >= 0.80:
        congestion = "HIGH"

    elif utilization >= 0.60:
        congestion = "MEDIUM"

    else:
        congestion = "LOW"

    network[source][destination]["congestion"] = congestion

# ============================================================
# 6. CALCULATE EFFECTIVE DELAY
# ============================================================

for source, destination in network.edges():

    base_delay = network[source][destination]["delay"]
    utilization = network[source][destination]["utilization"]

    effective_delay = base_delay * (1 + utilization)

    network[source][destination]["effective_delay"] = effective_delay

# ============================================================
# 7. DIJKSTRA ROUTING
# ============================================================

source_router = "R1"
destination_router = "R5"

best_path = nx.shortest_path(
    network,
    source=source_router,
    target=destination_router,
    weight="effective_delay"
)

# ============================================================
# 8. CALCULATE PATH METRICS
# ============================================================

total_delay = 0
path_bandwidths = []
total_traffic = 0

for i in range(len(best_path) - 1):

    source = best_path[i]
    destination = best_path[i + 1]

    link_data = network[source][destination]

    total_delay += link_data["effective_delay"]

    path_bandwidths.append(link_data["bandwidth"])

    total_traffic += link_data["traffic"]

# Bottleneck bandwidth
throughput = min(path_bandwidths)

# Simple packet-loss model
packet_loss = max(0, (total_traffic / len(path_bandwidths) - throughput)
                  / throughput * 100)

# ============================================================
# 9. DISPLAY NETWORK INFORMATION
# ============================================================

print("================================================")
print("       SMART ROUTE OPTIMIZATION SYSTEM")
print("================================================")

print("\nRouters:")

for router in network.nodes:
    print(router)

print("\nNetwork Links:")
print("------------------------------------------------")

for source, destination, data in network.edges(data=True):

    print(
        source,
        "->",
        destination,
        "| Delay:", data["delay"], "ms",
        "| Effective Delay:", round(data["effective_delay"], 2), "ms",
        "| Bandwidth:", data["bandwidth"], "Mbps",
        "| Traffic:", round(data["traffic"], 2), "Mbps",
        "| Utilization:", round(data["utilization"] * 100, 2), "%",
        "| Congestion:", data["congestion"]
    )

# ============================================================
# 10. DISPLAY ROUTING RESULT
# ============================================================

print("\n================================================")
print("             DIJKSTRA ROUTING")
print("================================================")

print("Source Router      :", source_router)
print("Destination Router :", destination_router)

print("\nBest Path:")
print(" -> ".join(best_path))

print("\nPerformance Metrics:")
print("Total Delay        :", round(total_delay, 2), "ms")
print("Throughput         :", round(throughput, 2), "Mbps")
print("Packet Loss        :", round(packet_loss, 2), "%")

# ============================================================
# 11. SAVE RESULTS TO CSV
# ============================================================

from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "dijkstra_results.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Source",
        "Destination",
        "Path",
        "Total Delay (ms)",
        "Throughput (Mbps)",
        "Packet Loss (%)"
    ])

    writer.writerow([
        source_router,
        destination_router,
        " -> ".join(best_path),
        round(total_delay, 2),
        round(throughput, 2),
        round(packet_loss, 2)
    ])

print("\nResults saved to:")
print("results/dijkstra_results.csv")

# ============================================================
# 12. DRAW NETWORK
# ============================================================

pos = nx.spring_layout(network, seed=42)

nx.draw(
    network,
    pos,
    with_labels=True,
    node_size=2000,
    font_size=12
)

# Display base delay
edge_labels = {}

for source, destination, data in network.edges(data=True):

    edge_labels[(source, destination)] = (
        f'{data["delay"]} ms'
    )

nx.draw_networkx_edge_labels(
    network,
    pos,
    edge_labels=edge_labels
)

# Highlight Dijkstra path
path_edges = list(zip(best_path, best_path[1:]))

nx.draw_networkx_edges(
    network,
    pos,
    edgelist=path_edges,
    width=4
)

plt.title("Dijkstra-Based Network Route")

plt.show()