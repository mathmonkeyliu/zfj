from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import GRID_SIZE, GridState
from environment import BombPlanesEnv, build_outcome_table


@dataclass
class ElimAgent:
    """
    排除法（minimax）：
    遍历所有未选择的格子 a，考虑三种可能反馈 v∈{0=MISS,1=BODY,2=HEAD}。
    对每个 v，过滤候选布局后统计“剩余机头分布(label)”的种类数。
    取三种可能中的最坏情况（最大值），选择能让该最坏情况最小的格子。

    终止规则（与 ID3/C4.5 一致）：
    - 当候选集中只剩 1 种机头分布(label) 时，直接点剩余机头格。
    """

    outcomes: np.ndarray  # (N,100) uint8 in {0,1,2}
    label_ids: np.ndarray  # (N,) int32 (head-config id)
    labels: list[tuple[tuple[int, int], ...]]  # id -> canonical heads

    @staticmethod
    def from_layouts(layouts: list[dict[str, Any]]) -> "ElimAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        return ElimAgent(outcomes=outcomes, label_ids=label_ids, labels=labels)

    def _best_action(self, cand_idx: np.ndarray, unshot: np.ndarray) -> int:
        """
        选取使得 worst_case_unique_labels 最小的动作。
        次级排序：expected_unique_labels 更小；再其次：HEAD 概率更高；最后：动作 id 更小。
        """
        y = self.label_ids[cand_idx]
        uniq_labels, inv = np.unique(y, return_inverse=True)
        m = int(uniq_labels.size)

        features = np.flatnonzero(unshot)
        best_a = int(features[0])
        best_worst = 1 << 30
        best_expected = float("inf")
        best_head_prob = -1.0

        inv64 = inv.astype(np.int64, copy=False)
        for a in features:
            col = self.outcomes[cand_idx, int(a)].astype(np.int64, copy=False)  # 0/1/2
            combo = col * m + inv64
            cont = np.bincount(combo, minlength=3 * m).reshape(3, m)  # counts per outcome x label
            outcome_counts = cont.sum(axis=1).astype(np.float64, copy=False)
            n = float(outcome_counts.sum())
            if n <= 0:
                continue

            uniq_per_outcome = (cont > 0).sum(axis=1).astype(np.float64, copy=False)
            worst = int(uniq_per_outcome.max())
            probs = outcome_counts / n
            expected = float((probs * uniq_per_outcome).sum())
            head_prob = float(probs[2])

            if (
                (worst < best_worst)
                or (worst == best_worst and expected < best_expected - 1e-12)
                or (worst == best_worst and np.isclose(expected, best_expected) and head_prob > best_head_prob + 1e-12)
                or (
                    worst == best_worst
                    and np.isclose(expected, best_expected)
                    and np.isclose(head_prob, best_head_prob)
                    and int(a) < best_a
                )
            ):
                best_worst = worst
                best_expected = expected
                best_head_prob = head_prob
                best_a = int(a)

        return best_a

    def play_one(self, env: BombPlanesEnv, *, layout: dict[str, Any], max_steps: int = 500) -> int:
        _, info = env.reset(layout=layout)
        steps = 0
        heads_hit = 0

        unshot = info["action_mask"].copy()
        cand_idx = np.arange(self.outcomes.shape[0], dtype=np.int32)
        shot: set[tuple[int, int]] = set()

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


