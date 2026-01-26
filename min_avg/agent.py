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

    # 排序：增益大优先，增益相同则击中机头概率大优先
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
    progress_every: int = 5000


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
    def __init__(self, outcomes: np.ndarray, label_ids: np.ndarray, labels: list[tuple[tuple[int, int], ...]], cfg: MinAvgConfig, global_policy: dict[str, int]):
        self.outcomes = outcomes
        self.label_ids = label_ids
        self.labels = labels
        self.cfg = cfg
        # 运行时缓存：避免重复计算相同棋盘状态的最优解
        # Key: board_hex, Value: (best_action, total_steps, total_count)
        # 注意：这里不再存储 policy，只存储数值评估结果
        self.memo: dict[str, tuple[int, int, int]] = {}
        # 全局唯一的策略字典，由外部传入
        self.policy = global_policy
        self._visited_nodes = 0

    def _progress(self) -> None:
        if not self.cfg.progress_enabled:
            return
        self._visited_nodes += 1
        if self._visited_nodes % self.cfg.progress_every == 0:
            print(f"[min_avg] visited nodes: {self._visited_nodes} | memo size: {len(self.memo)} | policy size: {len(self.policy)}")

    def search_state(self, board: np.ndarray, cand_idx: np.ndarray) -> tuple[int, int, int]:
        """
        状态节点搜索：
        1. 检查缓存 (memo)
        2. 检查终止条件
        3. 选 TopK 动作，调用 search_action
        4. 选最优动作，写入 global_policy
        5. 递归删除非最优动作的分支 (cleanup)
        6. 返回 (best_action, best_total_steps, total_count)
        """
        self._progress()
        
        # 1. 缓存检查
        key = _board_to_key(board)
        if key in self.memo:
            best_action, best_total_steps, total_count = self.memo[key]
            # 关键：如果命中了缓存，但该状态在 policy 中缺失（可能被其他分支误删），需要恢复
            if key not in self.policy and best_action >= 0:
                self.policy[key] = best_action
                # 注意：这里我们只恢复了当前节点的 policy，理想情况下应该递归恢复
                # 但由于我们有 memo，后续子节点访问时也会触发恢复逻辑
                # 或者我们可以主动调用 _ensure_branch_policy 来递归恢复，防止交互时查不到策略
                # 简单起见，这里只恢复当前节点。如果后续需要子节点策略，访问时自会恢复。
            return best_action, best_total_steps, total_count

        total_count = int(cand_idx.size)
        if total_count == 0:
            return -1, 0, 0

        # 2. 终止条件
        possible_labels = np.unique(self.label_ids[cand_idx])
        heads_hit = int((board == 3).sum())
        if heads_hit >= 3:
            return -1, 0, total_count

        if possible_labels.size == 1:
            heads = self.labels[int(possible_labels[0])]
            remaining = [(int(hx), int(hy)) for hx, hy in heads if board[int(hx), int(hy)] != 3]
            if not remaining:
                return -1, 0, total_count
            
            # 确定性策略
            temp_board = board.copy()
            sorted_remaining = sorted(remaining)
            steps_needed = len(sorted_remaining)
            total_steps = total_count * steps_needed
            
            # 写入这串必然的策略
            for i, (hx, hy) in enumerate(sorted_remaining):
                act = int(hx * GRID_SIZE + hy)
                pol_key = _board_to_key(temp_board)
                self.policy[pol_key] = act
                temp_board[hx, hy] = 3
            
            first_action = int(sorted_remaining[0][0] * GRID_SIZE + sorted_remaining[0][1])
            res = (first_action, int(total_steps), total_count)
            self.memo[key] = res
            return res

        unshot = board.reshape(-1) == 0
        if not unshot.any():
            return -1, 0, total_count

        # 3. 选择 TopK 动作
        ranked = _rank_actions_id3(self.outcomes, self.label_ids, cand_idx, unshot)
        ranked = _dedupe_rotations(ranked, GRID_SIZE)
        top_k = max(1, int(self.cfg.top_k))
        ranked = ranked[:top_k]
        
        if not ranked:
            return -1, 0, total_count

        best_action = -1
        best_avg = float("inf")
        best_total_steps = 0
        
        # 遍历动作
        for a in ranked:
            a = int(a)
            # 递归调用
            act_steps, act_count = self.search_action(board, cand_idx, a)
            
            current_total = act_steps + total_count
            avg = current_total / total_count if total_count > 0 else float("inf")

            if avg < best_avg:
                # 之前的 best_action 现在变成了“非最优”，需要清理它的分支
                if best_action != -1:
                    self._purge_branch(board, cand_idx, best_action)
                
                best_avg = avg
                best_action = a
                best_total_steps = current_total
            else:
                # 当前动作 a 不是最优，清理它的分支
                self._purge_branch(board, cand_idx, a)

        # 4. 记录最优策略
        if best_action != -1:
            self.policy[key] = best_action
        
        res = (best_action, int(best_total_steps), total_count)
        self.memo[key] = res
        return res

    def search_action(self, board: np.ndarray, cand_idx: np.ndarray, action: int) -> tuple[int, int]:
        """
        动作节点搜索：
        不再聚合 policy，只聚合数值结果。
        """
        total_steps = 0
        total_count = 0

        for obs_v in (0, 1, 2): # MISS, BODY, HEAD
            mask = self.outcomes[cand_idx, action] == obs_v
            if not mask.any():
                continue
            
            sub_cand_idx = cand_idx[mask]
            
            next_board = board.copy()
            next_board.reshape(-1)[action] = obs_v + 1
            
            _, sub_steps, sub_count = self.search_state(next_board, sub_cand_idx)
            
            total_steps += sub_steps
            total_count += sub_count

        return total_steps, total_count

    def _purge_branch(self, board: np.ndarray, cand_idx: np.ndarray, action: int) -> None:
        """
        递归删除指定动作产生的子策略。
        """
        for obs_v in (0, 1, 2):
            mask = self.outcomes[cand_idx, action] == obs_v
            if not mask.any():
                continue
            
            sub_cand_idx = cand_idx[mask]
            next_board = board.copy()
            next_board.reshape(-1)[action] = obs_v + 1
            
            key = _board_to_key(next_board)
            if key in self.policy:
                # 如果该子状态在策略中有记录，说明它是本分支生成的（或者共享的）
                # 这里我们假设它是本分支生成的（或者是共享的但现在父节点不要这个分支了）
                # 我们删除它，并递归删除它的子动作
                child_action = self.policy[key]
                del self.policy[key]
                self._purge_branch(next_board, sub_cand_idx, child_action)
            
            # 注意：不删除 memo，保留数值缓存以便可能的复用


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
            # 如果已有 policy，复用它的字典
            global_policy = self.policy.policy if self.policy else {}
            self._planner = MinAvgPlanner(self.outcomes, self.label_ids, self.labels, self.cfg, global_policy)
            if self.policy is None:
                 self.policy = MinAvgPolicy(policy=global_policy, grid_size=GRID_SIZE)
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
        # 注意：这里直接调用 search_state，它会进行递归搜索
        best_action, _, _ = planner.search_state(board, cand_idx)
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
