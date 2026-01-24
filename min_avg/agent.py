from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import GRID_SIZE, MIN_AVG_TOPK, GridState
from environment import BombPlanesEnv, build_outcome_table


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64, copy=False) / total
    return float(-(p * np.log2(p)).sum())


def _rank_actions_id3(
    outcomes: np.ndarray, label_ids: np.ndarray, cand_idx: np.ndarray, unshot: np.ndarray
) -> list[int]:
    y = label_ids[cand_idx]
    uniq, inv = np.unique(y, return_inverse=True)
    m = int(uniq.size)
    base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m).astype(np.float64, copy=False))

    features = np.flatnonzero(unshot)
    inv64 = inv.astype(np.int64, copy=False)

    scored: list[tuple[float, float, int]] = []
    for a in features:
        col = outcomes[cand_idx, int(a)].astype(np.int64, copy=False)
        combo = col * m + inv64
        cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
        outcome_counts = cont.sum(axis=1)
        n = int(outcome_counts.sum())
        if n == 0:
            continue

        cond_entropy = 0.0
        for v in range(3):
            nv = int(outcome_counts[v])
            if nv == 0:
                continue
            cond_entropy += (nv / n) * _entropy_from_counts(cont[v])

        gain = base_entropy - cond_entropy
        head_prob = float(outcome_counts[2] / n)
        scored.append((gain, head_prob, int(a)))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [a for _, __, a in scored]


def _action_rotations(action: int, grid_size: int) -> tuple[int, int, int, int]:
    x, y = divmod(int(action), grid_size)
    r90 = (y, grid_size - 1 - x)
    r180 = (grid_size - 1 - x, grid_size - 1 - y)
    r270 = (grid_size - 1 - y, x)
    return (
        int(action),
        r90[0] * grid_size + r90[1],
        r180[0] * grid_size + r180[1],
        r270[0] * grid_size + r270[1],
    )


def _dedupe_rotations(actions: list[int], grid_size: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for a in actions:
        key = min(_action_rotations(int(a), grid_size))
        if key in seen:
            continue
        seen.add(key)
        out.append(int(a))
    return out


def _board_to_key(board: np.ndarray) -> str:
    flat = board.astype(np.uint8, copy=False).ravel()
    return flat.tobytes().hex()


def _board_from_state(board_state: list[list[GridState]] | np.ndarray) -> np.ndarray:
    if isinstance(board_state, np.ndarray):
        if board_state.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"board_state shape must be {(GRID_SIZE, GRID_SIZE)}, got {board_state.shape}")
        return board_state.astype(np.uint8, copy=False)

    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cell = board_state[x][y]
            if cell == GridState.UNKNOWN:
                out[x, y] = 0
            elif cell == GridState.MISS:
                out[x, y] = 1
            elif cell == GridState.BODY:
                out[x, y] = 2
            elif cell == GridState.HEAD:
                out[x, y] = 3
            else:
                raise ValueError(f"Unknown GridState: {cell}")
    return out


@dataclass(frozen=True)
class MinAvgConfig:
    top_k: int = MIN_AVG_TOPK
    progress_enabled: bool = True
    progress_every: int = 2000


@dataclass
class MinAvgPolicy:
    policy: dict[str, int]
    grid_size: int = GRID_SIZE
    version: int = 1

    def save(self, path: str | Path) -> None:
        p = Path(path)
        payload = {"version": self.version, "grid_size": self.grid_size, "policy": self.policy}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "MinAvgPolicy":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "policy" not in data:
            raise ValueError(f"Invalid min_avg policy file: {path}")
        grid_size = int(data.get("grid_size", GRID_SIZE))
        version = int(data.get("version", 1))
        policy_raw = data["policy"]
        if not isinstance(policy_raw, dict):
            raise ValueError(f"Invalid policy mapping in {path}")
        policy = {str(k): int(v) for k, v in policy_raw.items()}
        return MinAvgPolicy(policy=policy, grid_size=grid_size, version=version)


class MinAvgPlanner:
    def __init__(self, outcomes: np.ndarray, label_ids: np.ndarray, labels: list[tuple[tuple[int, int], ...]], cfg: MinAvgConfig):
        self.outcomes = outcomes
        self.label_ids = label_ids
        self.labels = labels
        self.cfg = cfg
        self.cache: dict[str, tuple[int, int, int]] = {}
        self.policy: dict[str, int] = {}
        self._visited = 0

    def _progress(self) -> None:
        if not self.cfg.progress_enabled:
            return
        self._visited += 1
        if self._visited % self.cfg.progress_every == 0:
            print(f"[min_avg] visited states: {self._visited} | policy size: {len(self.policy)}")

    def search(self, board: np.ndarray, cand_idx: np.ndarray) -> tuple[int, int, int]:
        key = _board_to_key(board)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self._progress()

        total_count = int(cand_idx.size)
        if total_count == 0:
            result = (-1, 0, 0)
            self.cache[key] = result
            return result

        heads_hit = int((board == 3).sum())
        if heads_hit >= 3:
            result = (-1, 0, total_count)
            self.cache[key] = result
            return result

        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size == 1:
            heads = self.labels[int(possible_labels[0])]
            remaining = [(int(hx), int(hy)) for hx, hy in heads if board[int(hx), int(hy)] != 3]
            if not remaining:
                result = (-1, 0, total_count)
                self.cache[key] = result
                return result
            a = min(hx * GRID_SIZE + hy for hx, hy in remaining)
            total_steps = total_count * len(remaining)
            result = (a, total_steps, total_count)
            self.cache[key] = result
            self.policy[key] = int(a)
            return result

        unshot = board.reshape(-1) == 0
        if not unshot.any():
            result = (-1, 0, total_count)
            self.cache[key] = result
            return result

        ranked = _rank_actions_id3(self.outcomes, self.label_ids, cand_idx, unshot)
        ranked = _dedupe_rotations(ranked, GRID_SIZE)
        top_k = max(1, int(self.cfg.top_k))
        ranked = ranked[:top_k]
        if not ranked:
            result = (-1, 0, total_count)
            self.cache[key] = result
            return result

        best_action = int(ranked[0])
        best_total_steps = float("inf")

        for a in ranked:
            a = int(a)
            total_steps = 0
            for obs_v in (0, 1, 2):
                mask = self.outcomes[cand_idx, a] == obs_v
                if not mask.any():
                    continue
                child_idx = cand_idx[mask]
                child_board = board.copy()
                child_board.reshape(-1)[a] = obs_v + 1
                _, child_total_steps, _ = self.search(child_board, child_idx)
                total_steps += int(child_total_steps)
            total_steps += total_count
            if total_steps < best_total_steps:
                best_total_steps = float(total_steps)
                best_action = int(a)

        result = (best_action, int(best_total_steps), total_count)
        self.cache[key] = result
        self.policy[key] = int(best_action)
        return result


@dataclass
class MinAvgAgent:
    outcomes: np.ndarray
    label_ids: np.ndarray
    labels: list[tuple[tuple[int, int], ...]]
    cfg: MinAvgConfig
    policy: MinAvgPolicy | None = None
    _planner: MinAvgPlanner | None = None

    @staticmethod
    def from_layouts(layouts: list[dict[str, Any]], cfg: MinAvgConfig | None = None) -> "MinAvgAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        cfg = MinAvgConfig() if cfg is None else cfg
        return MinAvgAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg)

    @staticmethod
    def from_precomputed(
        precomputed_path: str | Path,
        layouts: list[dict[str, Any]],
        cfg: MinAvgConfig | None = None,
    ) -> "MinAvgAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        policy = MinAvgPolicy.load(precomputed_path)
        cfg = MinAvgConfig() if cfg is None else cfg
        return MinAvgAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg, policy=policy)

    def _planner_for_fallback(self) -> MinAvgPlanner:
        if self._planner is None:
            self._planner = MinAvgPlanner(self.outcomes, self.label_ids, self.labels, self.cfg)
        return self._planner

    def choose_action(
        self,
        *,
        board_state: list[list[GridState]] | np.ndarray,
        cand_idx: np.ndarray,
        unshot_actions: np.ndarray,
        heads_hit: int,
    ) -> int:
        board = _board_from_state(board_state)
        key = _board_to_key(board)
        if self.policy is not None and key in self.policy.policy:
            return int(self.policy.policy[key])

        planner = self._planner_for_fallback()
        best_action, _, _ = planner.search(board, cand_idx)
        if best_action >= 0:
            return int(best_action)

        ranked = _rank_actions_id3(self.outcomes, self.label_ids, cand_idx, unshot_actions)
        if not ranked:
            raise RuntimeError("No available actions to choose from.")
        return int(ranked[0])

    def play_one(self, env: BombPlanesEnv, *, layout: dict[str, Any], max_steps: int = 500) -> int:
        _, info = env.reset(layout=layout)
        steps = 0
        heads_hit = 0

        board = env.board.copy().astype(np.uint8, copy=False)
        unshot = info["action_mask"].copy()
        cand_idx = np.arange(self.outcomes.shape[0], dtype=np.int32)

        while heads_hit < 3 and steps < max_steps:
            if not unshot.any():
                break
            a = self.choose_action(board_state=board, cand_idx=cand_idx, unshot_actions=unshot, heads_hit=heads_hit)
            x, y = divmod(int(a), GRID_SIZE)
            unshot[int(a)] = False

            sr = env.step((x, y))
            steps += 1
            result = sr.info["result"]
            if result == GridState.HEAD:
                heads_hit += 1

            obs_v = 2 if result == GridState.HEAD else (1 if result == GridState.BODY else 0)
            board[x, y] = obs_v + 1

            col = self.outcomes[cand_idx, int(a)]
            cand_idx = cand_idx[col == obs_v]
            if cand_idx.size == 0:
                break

        return steps
