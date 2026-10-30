import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

# Use Member 1's shared environment as the single source of network topology.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from routing_environment import RoutingEnvironment
from dqn_agent import DQNAgent


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = 700
TARGET_UPDATE_EVERY = 10
SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = RoutingEnvironment(seed=SEED)
    agent = DQNAgent(env.state_size, env.action_size)

    rewards_history = []
    delay_history = []
    success_history = []
    loss_history = []

    print("Training DQN routing agent")
    print(f"Device: {agent.device}")
    print(f"State size: {env.state_size}; action size: {env.action_size}")
    print(f"Episodes: {EPISODES}\n")

    for episode in range(1, EPISODES + 1):
        state = env.reset(source=0, destination=4)
        total_reward = 0.0
        last_info = {}
        episode_losses = []
        done = False

        while not done:
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, valid_actions, training=True)

            next_state, reward, done, info = env.step(action)
            next_valid = [] if done else env.valid_actions()

            agent.remember(
                state, action, reward, next_state, done, next_valid
            )
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            last_info = info

        agent.decay_epsilon()
        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

        success = bool(last_info.get("success", False))
        delay = float(last_info.get("total_delay", 0.0))

        rewards_history.append(total_reward)
        delay_history.append(delay)
        success_history.append(int(success))
        loss_history.append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        if episode == 1 or episode % 50 == 0:
            start = max(0, len(rewards_history) - 50)
            avg_reward = np.mean(rewards_history[start:])
            avg_success = 100 * np.mean(success_history[start:])
            avg_delay = np.mean(delay_history[start:])
            print(
                f"Episode {episode:4d}/{EPISODES} | "
                f"avg reward (last 50): {avg_reward:7.2f} | "
                f"success (last 50): {avg_success:5.1f}% | "
                f"avg delay: {avg_delay:6.2f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    model_path = RESULTS_DIR / "dqn_model.pth"
    agent.save(str(model_path))

    csv_path = RESULTS_DIR / "training_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "total_delay", "success", "mean_loss"])
        for i in range(EPISODES):
            writer.writerow([
                i + 1,
                rewards_history[i],
                delay_history[i],
                success_history[i],
                loss_history[i],
            ])

    # Plot a moving average to make learning trends easier to see.
    window = 25
    if len(rewards_history) >= window:
        moving_rewards = np.convolve(
            rewards_history, np.ones(window) / window, mode="valid"
        )
        moving_success = np.convolve(
            success_history, np.ones(window) / window, mode="valid"
        ) * 100
    else:
        moving_rewards = rewards_history
        moving_success = np.asarray(success_history) * 100

    plt.figure()
    plt.plot(range(1, len(rewards_history) + 1), rewards_history, alpha=0.25,
             label="Episode reward")
    plt.plot(range(window, len(rewards_history) + 1), moving_rewards,
             label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN Training Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_rewards.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(range(1, len(success_history) + 1), np.asarray(success_history) * 100,
             alpha=0.25, label="Episode success")
    plt.plot(range(window, len(success_history) + 1), moving_success,
             label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reached destination (%)")
    plt.title("Routing Success During Training")
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_success.png", dpi=150)
    plt.close()

    print("\nTraining finished.")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {csv_path}")
    print(f"Saved plots: {RESULTS_DIR / 'training_rewards.png'}")
    print(f"Saved plots: {RESULTS_DIR / 'training_success.png'}")


if __name__ == "__main__":
    main()
