from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .constants import BOARD_SIZE, CELL_HEAD

State = np.ndarray
Action = int


class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Iterable[Transition]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class Agent:
    def __init__(
        self,
        device: torch.device,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.device = device
        input_dim = BOARD_SIZE * BOARD_SIZE
        output_dim = input_dim
        self.policy_net = DQN(input_dim, hidden_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, hidden_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            raise ValueError("No valid actions available, invalid state.")

        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_indices))

        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)
        mask_tensor = torch.from_numpy(mask).to(self.device)
        masked_q = q_values.masked_fill(mask_tensor == 0, -1e9)
        return int(torch.argmax(masked_q).item())

    def optimize(self, replay: ReplayBuffer, batch_size: int) -> float:
        batch = replay.sample(batch_size)
        state_batch = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        next_mask_batch = torch.tensor(np.stack([b.next_mask for b in batch]), dtype=torch.float32, device=self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state_batch)
            next_q = next_q.masked_fill(next_mask_batch == 0, -1e9)
            max_next_q = torch.max(next_q, dim=1).values
            target_q = reward_batch + (1 - done_batch) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def hard_update(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: Path, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict, strict=strict)
        self.hard_update()

    @staticmethod
    def preprocess_observation(obs: np.ndarray) -> np.ndarray:
        return obs.reshape(BOARD_SIZE * BOARD_SIZE)
