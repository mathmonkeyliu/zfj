from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import GRID_SIZE, GridState
from environment import BombPlanesEnv, build_outcome_table


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


@dataclass
class C45Agent:
    """
    Online greedy C4.5 (per-step choose the unshot cell with maximal gain ratio),
    where the class label is the 3-head configuration.

    Terminal rule (per requirement):
    - Once only one possible head-configuration remains, directly shoot remaining heads.
    """

    outcomes: np.ndarray  # (N,100) uint8 in {0,1,2}
    label_ids: np.ndarray  # (N,) int32 (head-config id)
    labels: list[tuple[tuple[int, int], ...]]  # id -> canonical heads

    @staticmethod
    def from_layouts(layouts: list[dict[str, Any]]) -> "C45Agent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        return C45Agent(outcomes=outcomes, label_ids=label_ids, labels=labels)

    def _best_action(self, cand_idx: np.ndarray, unshot: np.ndarray) -> int:
        """
        Choose feature maximizing gain_ratio = information_gain / split_info.
        """
        y = self.label_ids[cand_idx]
        uniq_labels, inv = np.unique(y, return_inverse=True)
        m = int(uniq_labels.size)
        base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m))

        features = np.flatnonzero(unshot)
        best_ratio = -1.0
        best_a = int(features[0])
        best_head_prob = -1.0

        inv64 = inv.astype(np.int64, copy=False)
        for a in features:
            col = self.outcomes[cand_idx, int(a)].astype(np.int64, copy=False)
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

            info_gain = base_entropy - cond_entropy
            split_info = _entropy_from_counts(outcome_counts.astype(np.float64))
            if split_info <= 0:
                continue
            ratio = float(info_gain / split_info)
            head_prob = float(outcome_counts[2] / n)

            if (ratio > best_ratio) or (np.isclose(ratio, best_ratio) and head_prob > best_head_prob):
                best_ratio = ratio
                best_a = int(a)
                best_head_prob = head_prob

        return best_a

    def play_one(self, env: BombPlanesEnv, *, layout: dict[str, Any], max_steps: int = 500) -> int:
        _, info = env.reset(layout=layout)
        steps = 0
        heads_hit = 0

        unshot = info["action_mask"].copy()
        cand_idx = np.arange(self.outcomes.shape[0], dtype=np.int32)
        shot = set()

        def shoot_xy(x: int, y: int) -> GridState:
            nonlocal steps, heads_hit
            sr = env.step((x, y))
            steps += 1
            shot.add((x, y))
            if sr.info["result"] == GridState.HEAD:
                heads_hit += 1
            return sr.info["result"]

        while heads_hit < 3 and steps < max_steps:
            possible_labels = np.unique(self.label_ids[cand_idx])
            if possible_labels.size == 1:
                heads = self.labels[int(possible_labels[0])]
                for hx, hy in heads:
                    if (int(hx), int(hy)) not in shot:
                        shoot_xy(int(hx), int(hy))
                        if heads_hit >= 3:
                            break
                break

            if not unshot.any():
                break

            a = self._best_action(cand_idx, unshot)
            x, y = divmod(int(a), GRID_SIZE)
            unshot[int(a)] = False

            result = shoot_xy(x, y)
            obs_v = 2 if result == GridState.HEAD else (1 if result == GridState.BODY else 0)

            col = self.outcomes[cand_idx, int(a)]
            cand_idx = cand_idx[col == obs_v]
            if cand_idx.size == 0:
                break

        return steps


