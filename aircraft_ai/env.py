from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .board import AttackOutcome, AttackResult, Board
from .constants import (
    BOARD_SIZE,
    CELL_HEAD,
    CELL_HIT,
    CELL_MISS,
    CELL_UNKNOWN,
    MAX_TURNS,
    REWARD_COMPLETE,
    REWARD_HEAD,
    REWARD_STEP,
)


@dataclass
class AttackFeedback:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class AttackState:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.knowledge = np.full((BOARD_SIZE, BOARD_SIZE), CELL_UNKNOWN, dtype=np.int8)
        self.turns = 0
        self.done = False

    def bind_board(self, board: Board) -> None:
        self.board = board
        self.reset_memory()

    def reset_memory(self) -> None:
        self.knowledge.fill(CELL_UNKNOWN)
        self.turns = 0
        self.done = False

    def _update_cell(self, row: int, col: int, outcome: AttackOutcome) -> None:
        if outcome == AttackOutcome.MISS:
            self.knowledge[row, col] = CELL_MISS
        elif outcome == AttackOutcome.HIT:
            self.knowledge[row, col] = CELL_HIT
        elif outcome == AttackOutcome.HEAD:
            self.knowledge[row, col] = CELL_HEAD

    def _obs(self) -> np.ndarray:
        return (self.knowledge.astype(np.float32) / float(CELL_HEAD)).reshape(-1)

    def observation(self) -> np.ndarray:
        return self._obs()

    def action_mask(self) -> np.ndarray:
        return (self.knowledge.reshape(-1) == CELL_UNKNOWN).astype(np.float32)

    def apply_action(self, action_index: int) -> AttackFeedback:
        if self.done:
            return AttackFeedback(self._obs(), 0.0, True, {"reason": "finished"})

        row = action_index // BOARD_SIZE
        col = action_index % BOARD_SIZE

        result = self.board.attack((col, row))
        reward = 0.0
        info = {"result": result}

        if self.knowledge[row, col] != CELL_UNKNOWN:
            reward += REWARD_STEP * 2
            return AttackFeedback(self._obs(), reward, self.done, info)

        self.turns += 1

        reward += REWARD_STEP
        if result.outcome == AttackOutcome.HEAD:
            reward += REWARD_HEAD
            if self.board.all_planes_destroyed():
                self.done = True
                reward += REWARD_COMPLETE

        if self.turns >= MAX_TURNS and not self.done:
            self.done = True

        self._update_cell(row, col, result.outcome)
        return AttackFeedback(self._obs(), reward, self.done, info)


class AircraftEnv:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.board = Board()
        self.state = AttackState(self.board)

    def reset(self) -> np.ndarray:
        self.board.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        self.state.bind_board(self.board)
        return self.state.observation()

    def step(self, action: int) -> AttackFeedback:
        return self.state.apply_action(action)

    def action_mask(self) -> np.ndarray:
        return self.state.action_mask()

    def observation(self) -> np.ndarray:
        return self.state.observation()
