
from __future__ import annotations

from typing import List, Optional, Tuple, Any
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from encoding import EncodedState
from action import Action, ActionEncoder


class ReplayBuffer:
    """
    Experience replay buffer
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q Network - outputs Q values for each action dimension
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def compute_action_q_value(q_values, action_vec, action_catalog):
    """
    Compute Q value for a complete action - sum Q values of action components
    """
    type_q = (q_values[:, :len(action_catalog.action_type_order)] * action_vec[:, :len(action_catalog.action_type_order)]).sum(dim=1)
    table_q = (q_values[:, action_catalog.table_offset:action_catalog.attr_offset] * action_vec[:, action_catalog.table_offset:action_catalog.attr_offset]).sum(dim=1)
    attr_q = (q_values[:, action_catalog.attr_offset:action_catalog.edge_offset] * action_vec[:, action_catalog.attr_offset:action_catalog.edge_offset]).sum(dim=1)
    edge_q = (q_values[:, action_catalog.edge_offset:] * action_vec[:, action_catalog.edge_offset:]).sum(dim=1)
    return type_q + table_q + attr_q + edge_q


class DQNAgent:
    """
    DQN Agent
    """
    def __init__(
        self,
        state_dim,
        action_encoder,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update=10,
        buffer_capacity=10000,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_encoder = action_encoder
        self.action_catalog = action_encoder.catalog
        self.action_dim = action_encoder.catalog.action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.update_count = 0

    def select_action(self, encoded_state, legal_actions, training=True):
        """
        Select action
        """
        state = encoded_state.vector

        if training and random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)

            best_q = -float('inf')
            best_action = None

            for action in legal_actions:
                action_vec = self.action_encoder.encode_action(action)
                action_vec_tensor = torch.FloatTensor(action_vec).unsqueeze(0).to(self.device)
                q_val = compute_action_q_value(q_values, action_vec_tensor, self.action_catalog)
                q_val = q_val.item()

                if q_val > best_q:
                    best_q = q_val
                    best_action = action

            action = best_action if best_action is not None else random.choice(legal_actions)

        action_vec = self.action_encoder.encode_action(action)
        return action, action_vec

    def update(self, batch_size=64):
        """
        Update network
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        state_q_values = self.policy_net(states_tensor)
        next_state_q_values = self.target_net(next_states_tensor)

        q_values = compute_action_q_value(state_q_values, actions_tensor, self.action_catalog)

        max_next_q = next_state_q_values.max(dim=1)[0]
        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        loss = F.mse_loss(q_values, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]

