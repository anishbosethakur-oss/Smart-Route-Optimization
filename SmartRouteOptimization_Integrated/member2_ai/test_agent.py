import random
from pathlib import Path

import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from routing_environment import RoutingEnvironment
from dqn_agent import DQNAgent
MODEL_PATH = ROOT / "results" / "dqn_model.pth"


def run_episode(env, agent, source=0, destination=4, training=False):
    state = env.reset(source=source, destination=destination)
    done = False
    total_reward = 0.0
    final_info = {}

    while not done:
        valid_actions = env.valid_actions()
        action = agent.choose_action(state, valid_actions, training=training)
        state, reward, done, final_info = env.step(action)
        total_reward += reward

    return env.path_names(), total_reward, final_info


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    if not MODEL_PATH.exists():
        print("Trained model not found.")
        print("Run this first from the project root:")
        print("  python member2_ai/train.py")
        return

    env = RoutingEnvironment(seed=123)
    agent = DQNAgent(env.state_size, env.action_size)
    agent.load(str(MODEL_PATH))
    agent.epsilon = 0.0  # Use learned values, not random exploration.

    print("Testing the trained DQN agent\n")
    trials = 30
    successes = 0
    delays = []

    for trial in range(1, trials + 1):
        path, reward, info = run_episode(env, agent, source=0, destination=4)
        success = bool(info.get("success", False))
        successes += int(success)
        delay = float(info.get("total_delay", 0.0))
        if success:
            delays.append(delay)

        status = "SUCCESS" if success else "FAILED"
        print(
            f"Trial {trial:02d}: {status:7s} | "
            f"Path: {' -> '.join(path)} | "
            f"Delay: {delay:.2f} | Reward: {reward:.2f}"
        )

    print("\nSummary")
    print(f"Successful routes: {successes}/{trials} ({100 * successes / trials:.1f}%)")
    if delays:
        print(f"Mean delay of successful routes: {np.mean(delays):.2f}")
    else:
        print("No successful routes in these trials.")
    print("\nNote: traffic changes each trial, so results can vary.")


if __name__ == "__main__":
    main()
