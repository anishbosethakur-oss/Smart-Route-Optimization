import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """Neural network that estimates Q-values for each possible next router."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """DQN agent with experience replay, a target network, and epsilon-greedy actions."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.992,
        batch_size=64,
        memory_size=10000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()

    def choose_action(self, state, valid_actions, training=True):
        valid_actions = list(valid_actions)
        if not valid_actions:
            raise ValueError("No valid neighboring routers were supplied.")

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)[0]

        return max(valid_actions, key=lambda a: q_values[int(a)].item())

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            tuple(int(a) for a in next_valid_actions),
        ))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = torch.as_tensor(
            np.asarray([x[0] for x in batch], dtype=np.float32),
            device=self.device
        )
        actions = torch.as_tensor(
            [x[1] for x in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [x[2] for x in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray([x[3] for x in batch], dtype=np.float32),
            device=self.device
        )
        dones = torch.as_tensor(
            [x[4] for x in batch], dtype=torch.float32, device=self.device
        )

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            next_q_all = self.target_model(next_states)
            for i, item in enumerate(batch):
                if not item[4] and item[5]:
                    valid = list(item[5])
                    next_q[i] = next_q_all[i, valid].max()
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_function(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename="results/dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="results/dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target_network()
