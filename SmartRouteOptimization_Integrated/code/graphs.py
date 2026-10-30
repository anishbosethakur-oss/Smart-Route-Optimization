import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SMART ROUTE OPTIMIZATION SYSTEM
# Day 5 - Performance Graphs
# ============================================================

# Read experiment results
data = pd.read_csv("results/dijkstra_experiments.csv")

# ============================================================
# 1. DISPLAY DATA
# ============================================================

print("================================================")
print("        DIJKSTRA EXPERIMENT RESULTS")
print("================================================")

print(data)

# ============================================================
# 2. CALCULATE AVERAGES
# ============================================================

average_delay = data["Total Delay (ms)"].mean()
average_throughput = data["Throughput (Mbps)"].mean()

print("\n================================================")
print("             AVERAGE PERFORMANCE")
print("================================================")

print(
    "Average Delay:",
    round(average_delay, 2),
    "ms"
)

print(
    "Average Throughput:",
    round(average_throughput, 2),
    "Mbps"
)

# ============================================================
# 3. DELAY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    data["Experiment"],
    data["Total Delay (ms)"],
    marker="o"
)

plt.xlabel("Experiment Number")
plt.ylabel("Total Delay (ms)")
plt.title("Dijkstra Routing - Delay Across Experiments")

plt.grid(True)

plt.savefig("results/dijkstra_delay_graph.png")

plt.show()

# ============================================================
# 4. THROUGHPUT GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    data["Experiment"],
    data["Throughput (Mbps)"],
    marker="o"
)

plt.xlabel("Experiment Number")
plt.ylabel("Throughput (Mbps)")
plt.title("Dijkstra Routing - Throughput Across Experiments")

plt.grid(True)

plt.savefig("results/dijkstra_throughput_graph.png")

plt.show()

# ============================================================
# 5. SUMMARY GRAPH
# ============================================================

metrics = [
    "Average Delay",
    "Average Throughput"
]

values = [
    average_delay,
    average_throughput
]

plt.figure(figsize=(8, 5))

plt.bar(metrics, values)

plt.ylabel("Value")
plt.title("Dijkstra Routing - Average Performance")

plt.savefig("results/dijkstra_average_performance.png")

plt.show()

print("\nGraphs saved successfully!")
print("Check the results folder.")