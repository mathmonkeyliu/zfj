"""
Reinforcement-learning style environment for the "炸飞机" (Bombing Planes) game.

This repo models the *attacker* side: the environment samples (or is given) a fixed opponent
layout (3 planes). The agent chooses a grid cell to attack; the env returns one of:
- MISS: not a head, not a body
- BODY: plane body
- HEAD: plane head (a plane is "down", but no extra cells are revealed)

Episode terminates when all 3 heads are hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from config import GRID_SIZE, GridState, LAYOUT_FILE


OutcomeInt = Literal[0, 1, 2]  # 0=MISS, 1=BODY, 2=HEAD


def _to_action_xy(action: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(action, int):
        if not (0 <= action < GRID_SIZE * GRID_SIZE):
            raise ValueError(f"action int out of range: {action}")
        return divmod(action, GRID_SIZE)  # (row, col)
    x, y = action
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        raise ValueError(f"action (x,y) out of range: {(x, y)}")
    return x, y


def xy_to_action(x: int, y: int) -> int:
    """(x,y) where x=row, y=col -> discrete action id in [0,99]."""
    return x * GRID_SIZE + y


def normalize_head_label(heads: list[list[int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Canonicalize a 3-head configuration as a sorted tuple of (x,y)."""
    hs = [(int(x), int(y)) for x, y in heads]
    return tuple(sorted(hs))


def load_layouts(file_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load layouts from a jsonl file (one layout per line).
    Also accepts a json file that contains a list of layouts.
    """
    p = Path(file_path) if file_path is not None else Path(LAYOUT_FILE)
    if not p.exists():
        # fallback: some docs refer to "all_layouts.jsonl"
        alt = p.with_name("all_layouts.jsonl")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Layout file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        layouts: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                layouts.append(json.loads(line))
        return layouts

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {p}, got {type(data)}")
        return data

    raise ValueError(f"Unsupported layout file type: {p}")


def build_outcome_table(layouts: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[tuple[int, int], ...]]]:
    """
    Precompute per-layout outcome for each of 100 cells.

    Returns:
    - outcomes: (N, 100) uint8, values in {0=MISS,1=BODY,2=HEAD}
    - label_ids: (N,) int32 label id for head-configuration classification
    - labels: list[id] -> canonical 3-head tuple
    """
    n = len(layouts)
    outcomes = np.zeros((n, GRID_SIZE * GRID_SIZE), dtype=np.uint8)

    # label: head configuration only (ignore directions/bodies), per user requirement.
    # Use pure-Python interning to avoid numpy object-unique edge cases.
    label_to_id: dict[tuple[tuple[int, int], ...], int] = {}
    labels: list[tuple[tuple[int, int], ...]] = []
    label_ids = np.empty((n,), dtype=np.int32)
    for i, layout in enumerate(layouts):
        key = normalize_head_label(layout["heads"])
        lid = label_to_id.get(key)
        if lid is None:
            lid = len(labels)
            label_to_id[key] = lid
            labels.append(key)
        label_ids[i] = lid

    for i, layout in enumerate(layouts):
        for x, y in layout["bodies"]:
            outcomes[i, xy_to_action(int(x), int(y))] = 1
        for x, y in layout["heads"]:
            outcomes[i, xy_to_action(int(x), int(y))] = 2

    return outcomes, label_ids, labels


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BombPlanesEnv:
    """
    Minimal RL environment (gymnasium-like, but no external dependency).

    - Action space: 100 discrete actions (int in [0,99]) or (x,y)
    - Observation: int8 (10,10) board with values:
        0 UNKNOWN, 1 MISS, 2 BODY, 3 HEAD
    - `info["action_mask"]`: bool (100,) valid actions (not yet clicked)
    """

    def __init__(
        self,
        layouts: list[dict[str, Any]] | None = None,
        layout_file: str | Path | None = None,
        reward_mode: Literal["sparse", "dense"] = "sparse",
        illegal_action: Literal["raise", "penalize"] = "raise",
        max_steps: int = 200,
    ):
        self.layouts = layouts if layouts is not None else load_layouts(layout_file)
        self.reward_mode = reward_mode
        self.illegal_action = illegal_action
        self.max_steps = max_steps

        self._rng = np.random.default_rng()
        self._layout: dict[str, Any] | None = None

        self.board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)  # 0 unknown
        self._shot = np.zeros((GRID_SIZE * GRID_SIZE,), dtype=bool)
        self.steps = 0
        self.heads_hit = 0

    def seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, layout_index: int | None = None, layout: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        if layout is not None:
            self._layout = layout
        else:
            if layout_index is None:
                layout_index = int(self._rng.integers(0, len(self.layouts)))
            self._layout = self.layouts[layout_index]

        self.board.fill(0)
        self._shot.fill(False)
        self.steps = 0
        self.heads_hit = 0

        return self.board.copy(), {"action_mask": self.action_mask(), "layout_index": layout_index}

    def action_mask(self) -> np.ndarray:
        return (~self._shot).copy()

    def _result_at(self, x: int, y: int) -> GridState:
        assert self._layout is not None
        target = [x, y]
        if target in self._layout["heads"]:
            return GridState.HEAD
        if target in self._layout["bodies"]:
            return GridState.BODY
        return GridState.MISS

    def step(self, action: int | tuple[int, int]) -> StepResult:
        if self._layout is None:
            raise RuntimeError("Call reset() before step().")

        if self.steps >= self.max_steps:
            return StepResult(self.board.copy(), 0.0, False, True, {"action_mask": self.action_mask()})

        x, y = _to_action_xy(action)
        a = xy_to_action(x, y)

        if self._shot[a]:
            if self.illegal_action == "raise":
                raise ValueError(f"Illegal action: already shot {(x, y)}")
            # penalize: no state change
            self.steps += 1
            return StepResult(self.board.copy(), -1.0, False, False, {"illegal_action": True, "action_mask": self.action_mask()})

        self._shot[a] = True
        self.steps += 1

        result = self._result_at(x, y)
        if result == GridState.MISS:
            self.board[x, y] = 1
        elif result == GridState.BODY:
            self.board[x, y] = 2
        elif result == GridState.HEAD:
            self.board[x, y] = 3
            self.heads_hit += 1

        terminated = self.heads_hit >= 3
        truncated = self.steps >= self.max_steps and not terminated

        if self.reward_mode == "sparse":
            reward = float(result == GridState.HEAD)  # 1 only when hitting a head
        else:
            # dense: small step cost, reward for informative hits
            reward = -0.01 + (0.2 if result == GridState.BODY else 0.0) + (1.0 if result == GridState.HEAD else 0.0)

        info = {
            "result": result,
            "heads_hit": self.heads_hit,
            "steps": self.steps,
            "action_mask": self.action_mask(),
        }
        return StepResult(self.board.copy(), reward, terminated, truncated, info)


