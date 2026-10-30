import networkx as nx
import random
import csv

# ============================================================
# SMART ROUTE OPTIMIZATION SYSTEM
# Day 5 - Multiple Network Experiments
# ============================================================

# Number of experiments
NUMBER_OF_EXPERIMENTS = 20

# Source and destination
SOURCE = "R1"
DESTINATION = "R5"


# ============================================================
# FUNCTION TO CREATE AND SIMULATE NETWORK
# ============================================================

def run_experiment():

    # Create network
    network = nx.Graph()

    # Add routers
    routers = ["R1", "R2", "R3", "R4", "R5", "R6"]
    network.add_nodes_from(routers)

    # Add links
    network.add_edge("R1", "R2", delay=10, bandwidth=100)
    network.add_edge("R1", "R3", delay=15, bandwidth=100)
    network.add_edge("R2", "R4", delay=10, bandwidth=80)
    network.add_edge("R3", "R4", delay=20, bandwidth=100)
    network.add_edge("R3", "R5", delay=10, bandwidth=50)
    network.add_edge("R4", "R5", delay=5, bandwidth=100)
    network.add_edge("R2", "R6", delay=12, bandwidth=70)
    network.add_edge("R6", "R5", delay=8, bandwidth=80)

    # ========================================================
    # Generate dynamic utilization
    # ========================================================

    for source, destination in network.edges():

        utilization = random.uniform(0.20, 0.90)

        network[source][destination]["utilization"] = utilization

    # ========================================================
    # Calculate traffic
    # ========================================================

    for source, destination in network.edges():

        bandwidth = network[source][destination]["bandwidth"]
        utilization = network[source][destination]["utilization"]

        traffic = bandwidth * utilization

        network[source][destination]["traffic"] = traffic

    # ========================================================
    # Detect congestion
    # ========================================================

    for source, destination in network.edges():

        utilization = network[source][destination]["utilization"]

        if utilization >= 0.80:
            congestion = "HIGH"

        elif utilization >= 0.60:
            congestion = "MEDIUM"

        else:
            congestion = "LOW"

        network[source][destination]["congestion"] = congestion

    # ========================================================
    # Calculate effective delay
    # ========================================================

    for source, destination in network.edges():

        base_delay = network[source][destination]["delay"]
        utilization = network[source][destination]["utilization"]

        effective_delay = base_delay * (1 + utilization)

        network[source][destination]["effective_delay"] = effective_delay

    # ========================================================
    # Dijkstra routing
    # ========================================================

    best_path = nx.shortest_path(
        network,
        source=SOURCE,
        target=DESTINATION,
        weight="effective_delay"
    )

    # ========================================================
    # Calculate performance metrics
    # ========================================================

    total_delay = 0
    path_bandwidths = []

    for i in range(len(best_path) - 1):

        source = best_path[i]
        destination = best_path[i + 1]

        link_data = network[source][destination]

        total_delay += link_data["effective_delay"]

        path_bandwidths.append(link_data["bandwidth"])

    # Bottleneck bandwidth = minimum bandwidth on path
    throughput = min(path_bandwidths)

    # ========================================================
    # Return results
    # ========================================================

    return (
        " -> ".join(best_path),
        total_delay,
        throughput
    )


# ============================================================
# RUN MULTIPLE EXPERIMENTS
# ============================================================

results = []

print("================================================")
print("     RUNNING NETWORK EXPERIMENTS")
print("================================================")

for experiment_number in range(1, NUMBER_OF_EXPERIMENTS + 1):

    path, delay, throughput = run_experiment()

    results.append([
        experiment_number,
        SOURCE,
        DESTINATION,
        path,
        round(delay, 2),
        round(throughput, 2)
    ])

    print(
        "Experiment", experiment_number,
        "| Path:", path,
        "| Delay:", round(delay, 2), "ms",
        "| Throughput:", round(throughput, 2), "Mbps"
    )


# ============================================================
# SAVE ALL RESULTS
# ============================================================

with open(
    "results/dijkstra_experiments.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Experiment",
        "Source",
        "Destination",
        "Path",
        "Total Delay (ms)",
        "Throughput (Mbps)"
    ])

    writer.writerows(results)


# ============================================================
# CALCULATE AVERAGES
# ============================================================

average_delay = sum(
    row[4] for row in results
) / NUMBER_OF_EXPERIMENTS

average_throughput = sum(
    row[5] for row in results
) / NUMBER_OF_EXPERIMENTS


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n================================================")
print("          EXPERIMENT SUMMARY")
print("================================================")

print(
    "Number of Experiments :",
    NUMBER_OF_EXPERIMENTS
)

print(
    "Average Delay         :",
    round(average_delay, 2),
    "ms"
)

print(
    "Average Throughput    :",
    round(average_throughput, 2),
    "Mbps"
)

print("\nResults saved to:")
print("results/dijkstra_experiments.csv")