from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import GRID_SIZE, GridState
from environment import BombPlanesEnv, build_outcome_table
from monkey.monkey_config import MonkeyConfig
from monkey.search import SearchContext, make_progress, solve_minimax_and_record_best_action
from monkey.state_hash import state_hash_hex
from monkey.symmetry import SymmetryGroup
from monkey.search import _pick_topk_actions  # type: ignore


@dataclass
class MonkeyAgent:
    """
    MiniMax + AlphaBeta + ID3 TopK 的策略。

    两种用法：
    - 在线：每一步基于当前 cand_idx/unshot/shots 做 minimax
    - 预计算：从 JSON 表 O(1) 查“该状态最优下一步动作”
    """

    outcomes: np.ndarray
    label_ids: np.ndarray
    labels: list[tuple[tuple[int, int], ...]]
    cfg: MonkeyConfig

    # precomputed policy: state_hash -> action_id
    precomputed: dict[str, int] | None = None

    # interactive state (shots history)
    _shots: dict[int, int] | None = None

    @staticmethod
    def from_layouts(layouts: list[dict[str, Any]], *, cfg: MonkeyConfig | None = None) -> "MonkeyAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        return MonkeyAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=MonkeyConfig() if cfg is None else cfg)

    @staticmethod
    def from_precomputed(file_path: str | Path, layouts: list[dict[str, Any]]) -> "MonkeyAgent":
        p = Path(file_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "policy" not in data:
            raise ValueError("Invalid precomputed file: expected a dict with key 'policy'.")
        policy_raw = data["policy"]
        if not isinstance(policy_raw, dict):
            raise ValueError("Invalid precomputed file: 'policy' must be a dict.")
        policy: dict[str, int] = {}
        for k, v in policy_raw.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, list) and len(v) == 2:
                x, y = int(v[0]), int(v[1])
                policy[k] = x * GRID_SIZE + y
            else:
                policy[k] = int(v)

        outcomes, label_ids, labels = build_outcome_table(layouts)
        # cfg 可从文件里读；若不存在则用默认
        cfg = MonkeyConfig()
        cfg_raw = data.get("config")
        if isinstance(cfg_raw, dict):
            try:
                cfg = MonkeyConfig(**cfg_raw)  # type: ignore[arg-type]
            except TypeError:
                cfg = MonkeyConfig()
        return MonkeyAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg, precomputed=policy)

    def reset_session(self) -> None:
        self._shots = {}

    def observe(self, action_id: int, outcome: int) -> None:
        if self._shots is None:
            self._shots = {}
        self._shots[int(action_id)] = int(outcome)

    def choose_action(
        self,
        *,
        cand_idx: np.ndarray,
        unshot_actions: np.ndarray,
        heads_hit: int,
        shots: dict[int, int] | None = None,
    ) -> int:
        """
        返回 action_id (0..99)。
        - shots: 当前已观测到的格子->结果；若不提供，使用内部记录（interactive 会通过 observe() 更新）。
        """
        _ = heads_hit  # heads_hit 可由 shots 推导，这里保留参数以兼容旧调用

        shots_use = self._shots if shots is None else shots
        if shots_use is None:
            shots_use = {}

        # topk=1 时完全等价于 ID3（每层 MIN 只有一个动作可选），没必要跑 minimax。
        # 直接按信息增益选择当前一步即可：更快，且能保证与 id3/agent.py 的每步选择一致。
        if self.precomputed is None and int(self.cfg.top_k) == 1 and int(self.cfg.top_k_when_small_candidates) == 1:
            sym = SymmetryGroup.build()
            progress = make_progress(self.cfg, enabled=False)
            ctx = SearchContext(
                outcomes=self.outcomes,
                label_ids=self.label_ids,
                labels=self.labels,
                cfg=self.cfg,
                sym=sym,
                progress=progress,
                value_cache={},
                best_action_cache={},
            )
            acts = _pick_topk_actions(ctx, cand_idx, unshot_actions, dict(shots_use))
            if acts:
                return int(acts[0])

        # 如果有预计算表，优先 O(1) 查表
        if self.precomputed is not None:
            h = state_hash_hex(shots_use)
            a = self.precomputed.get(h)
            if a is not None and bool(unshot_actions[int(a)]):
                return int(a)

        # 在线 minimax
        sym = SymmetryGroup.build()
        progress = make_progress(self.cfg, enabled=bool(self.cfg.progress_enabled))
        ctx = SearchContext(
            outcomes=self.outcomes,
            label_ids=self.label_ids,
            labels=self.labels,
            cfg=self.cfg,
            sym=sym,
            progress=progress,
            value_cache={},
            best_action_cache={},
        )
        try:
            solve_minimax_and_record_best_action(ctx, cand_idx, unshot_actions, dict(shots_use), depth=0)
        finally:
            progress.done()
        h = state_hash_hex(shots_use)
        a = ctx.best_action_cache.get(h)
        if a is None:
            # 回退：选第一个可用动作
            feats = np.flatnonzero(unshot_actions)
            return int(feats[0])
        return int(a)

    def play_one(self, env: BombPlanesEnv, *, layout: dict[str, Any], max_steps: int = 500) -> int:
        _, info = env.reset(layout=layout)
        steps = 0
        heads_hit = 0

        unshot = info["action_mask"].copy()
        cand_idx = np.arange(self.outcomes.shape[0], dtype=np.int32)
        self.reset_session()

        shot_xy: set[tuple[int, int]] = set()

        def shoot_xy(x: int, y: int) -> GridState:
            nonlocal steps, heads_hit
            sr = env.step((x, y))
            steps += 1
            shot_xy.add((x, y))
            if sr.info["result"] == GridState.HEAD:
                heads_hit += 1
            return sr.info["result"]

        while heads_hit < 3 and steps < max_steps:
            possible_labels = np.unique(self.label_ids[cand_idx])
            if possible_labels.size == 1:
                heads = self.labels[int(possible_labels[0])]
                for hx, hy in heads:
                    if (int(hx), int(hy)) not in shot_xy:
                        res = shoot_xy(int(hx), int(hy))
                        a = int(hx) * GRID_SIZE + int(hy)
                        obs_v = 2 if res == GridState.HEAD else (1 if res == GridState.BODY else 0)
                        self.observe(a, obs_v)
                        unshot[int(a)] = False
                        col = self.outcomes[cand_idx, int(a)]
                        cand_idx = cand_idx[col == obs_v]
                        if heads_hit >= 3:
                            break
                break

            if not unshot.any():
                break

            a = self.choose_action(cand_idx=cand_idx, unshot_actions=unshot, heads_hit=heads_hit)
            x, y = divmod(int(a), GRID_SIZE)
            unshot[int(a)] = False

            result = shoot_xy(int(x), int(y))
            obs_v = 2 if result == GridState.HEAD else (1 if result == GridState.BODY else 0)
            self.observe(int(a), obs_v)

            col = self.outcomes[cand_idx, int(a)]
            cand_idx = cand_idx[col == obs_v]
            if cand_idx.size == 0:
                break

        return int(steps)


