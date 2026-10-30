"""Compare greedy DQN with adaptive Dijkstra on matched traffic seeds.
Run after training the six-router DQN model.
"""
import csv
import random
from pathlib import Path
import sys

import numpy as np
import torch
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from routing_environment import RoutingEnvironment
from dqn_agent import DQNAgent

RESULTS = ROOT / "results"
MODEL = RESULTS / "dqn_model.pth"
TRIALS = 100
SOURCE, DESTINATION = 0, 4  # R1 to R5; R6 adds an alternative path


def run_dqn(seed, agent):
    env = RoutingEnvironment(seed=seed)
    state = env.reset(source=SOURCE, destination=DESTINATION)
    done = False
    total_reward = 0.0
    last_info = {}
    while not done:
        action = agent.choose_action(state, env.valid_actions(), training=False)
        state, reward, done, last_info = env.step(action)
        total_reward += reward
    return env.path_names(), float(last_info.get("total_delay", 0.0)), bool(last_info.get("success")), total_reward


def run_adaptive_dijkstra(seed):
    env = RoutingEnvironment(seed=seed)
    env.reset(source=SOURCE, destination=DESTINATION)
    done = False
    total_reward = 0.0
    last_info = {}
    while not done:
        current_name = env.id_to_router[env.current_router]
        destination_name = env.id_to_router[env.destination]
        # Recompute shortest path using current effective link delays at every hop.
        path = nx.shortest_path(env.network, current_name, destination_name, weight="effective_delay")
        next_name = path[1]
        action = env.router_to_id[next_name]
        _, reward, done, last_info = env.step(action)
        total_reward += reward
    return env.path_names(), float(last_info.get("total_delay", 0.0)), bool(last_info.get("success")), total_reward



def run_static_dijkstra(seed):
    """Conventional snapshot baseline: calculate one shortest path at source and follow it."""
    env = RoutingEnvironment(seed=seed)
    env.reset(source=SOURCE, destination=DESTINATION)
    current_name = env.id_to_router[env.current_router]
    destination_name = env.id_to_router[env.destination]
    fixed_path = nx.shortest_path(env.network, current_name, destination_name, weight="effective_delay")
    fixed_ids = [env.router_to_id[name] for name in fixed_path]
    done = False
    total_reward = 0.0
    last_info = {}
    hop = 1
    while not done:
        if hop >= len(fixed_ids):
            # Should only happen if an earlier step ended; protect against indexing.
            break
        action = fixed_ids[hop]
        _, reward, done, last_info = env.step(action)
        total_reward += reward
        hop += 1
    return env.path_names(), float(last_info.get("total_delay", 0.0)), bool(last_info.get("success")), total_reward


def main():
    if not MODEL.exists():
        print("No six-router model found. Train it first: python member2_ai/train.py")
        raise SystemExit(1)
    torch.manual_seed(123)
    np.random.seed(123)
    agent_env = RoutingEnvironment(seed=1)
    agent = DQNAgent(agent_env.state_size, agent_env.action_size)
    agent.load(str(MODEL))
    agent.epsilon = 0.0

    rows = []
    for trial in range(1, TRIALS + 1):
        seed = 1000 + trial
        for method, runner in (("DQN", lambda: run_dqn(seed, agent)),
                               ("Static Dijkstra", lambda: run_static_dijkstra(seed)),
                               ("Adaptive Dijkstra", lambda: run_adaptive_dijkstra(seed))):
            path, delay, success, reward = runner()
            rows.append([trial, seed, method, " -> ".join(path), round(delay, 4), int(success), round(reward, 4)])

    out = RESULTS / "dqn_vs_dijkstra.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "traffic_seed", "method", "path", "total_delay_ms", "success", "total_reward"])
        writer.writerows(rows)

    print(f"Paired comparison: {TRIALS} traffic seeds per method")
    for method in ("DQN", "Static Dijkstra", "Adaptive Dijkstra"):
        subset = [r for r in rows if r[2] == method]
        success_rate = 100 * sum(r[5] for r in subset) / len(subset)
        successful_delays = [r[4] for r in subset if r[5]]
        mean_delay = float(np.mean(successful_delays)) if successful_delays else float("nan")
        mean_reward = float(np.mean([r[6] for r in subset]))
        print(f"{method}: success={success_rate:.1f}%, mean successful delay={mean_delay:.2f} ms, mean reward={mean_reward:.2f}")
    print(f"Detailed results saved to {out}")
    print("Static Dijkstra fixes the route using costs observed at the source; adaptive Dijkstra recomputes the route at every hop.")
    print("Interpretation: lower delay and higher success/reward are generally preferable. Use the measured output; do not assume DQN wins.")


if __name__ == "__main__":
    main()
