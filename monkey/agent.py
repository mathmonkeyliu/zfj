"""
Monkey agent: minimax + alpha-beta pruning with ID3-based action selection.

玩家扮演 min（最小化步数），假想角色扮演 max（最大化步数）。
使用 ID3 方法中熵减最大的 top-k 个格子作为玩家的候选动作。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import GRID_SIZE, GridState
from environment import BombPlanesEnv, build_outcome_table, load_layouts

from .config import MonkeyConfig


def _entropy_from_counts(counts: np.ndarray) -> float:
    """计算熵"""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64, copy=False) / total
    return float(-(p * np.log2(p)).sum())


def _compute_entropy_gains(
    outcomes: np.ndarray,
    label_ids: np.ndarray,
    cand_idx: np.ndarray,
    unshot: np.ndarray,
) -> tuple[list[int], list[float]]:
    """
    计算所有未打击格子的信息增益（熵减）。
    
    返回：
    - actions: 动作列表（按熵减从大到小排序）
    - gains: 对应的熵减列表
    """
    y = label_ids[cand_idx]
    uniq_labels, inv = np.unique(y, return_inverse=True)
    m = int(uniq_labels.size)
    base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m).astype(np.float64, copy=False))

    features = np.flatnonzero(unshot)
    gains: list[float] = []
    actions: list[int] = []

    inv64 = inv.astype(np.int64, copy=False)
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
        gains.append(float(gain))
        actions.append(int(a))

    # 按熵减从大到小排序
    sorted_indices = sorted(range(len(gains)), key=lambda i: gains[i], reverse=True)
    sorted_actions = [actions[i] for i in sorted_indices]
    sorted_gains = [gains[i] for i in sorted_indices]

    return sorted_actions, sorted_gains


def _compute_entropy_gains_for_action(
    outcomes: np.ndarray,
    label_ids: np.ndarray,
    cand_idx: np.ndarray,
    action: int,
) -> tuple[list[float], list[int]]:
    """
    计算某个动作的三种可能结果（MISS=0, BODY=1, HEAD=2）的熵减。
    
    返回：
    - gains: 三种结果的熵减列表
    - outcome_orders: 按熵减从小到大排序的结果值列表（max 玩家的选择顺序）
    """
    y = label_ids[cand_idx]
    uniq_labels, inv = np.unique(y, return_inverse=True)
    m = int(uniq_labels.size)
    base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m).astype(np.float64, copy=False))

    col = outcomes[cand_idx, int(action)].astype(np.int64, copy=False)
    inv64 = inv.astype(np.int64, copy=False)
    combo = col * m + inv64
    cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
    outcome_counts = cont.sum(axis=1)

    n = int(outcome_counts.sum())
    gains: list[float] = []
    
    for v in range(3):
        nv = int(outcome_counts[v])
        if nv == 0:
            # 这个结果不可能发生
            gains.append(float('inf'))  # 无穷大表示不可能
        else:
            cond_entropy = _entropy_from_counts(cont[v])
            gain = base_entropy - cond_entropy
            gains.append(float(gain))

    # max 玩家按熵减从小到大排序（选择剩余熵最大的）
    outcome_orders = sorted(range(3), key=lambda i: gains[i])
    
    return gains, outcome_orders


@dataclass
class MonkeyAgent:
    """
    Monkey agent using minimax + alpha-beta pruning.
    
    玩家（min）试图最小化步数，假想对手（max）试图最大化步数。
    """

    outcomes: np.ndarray  # (N,100) uint8 in {0,1,2}
    label_ids: np.ndarray  # (N,) int32 (head-config id)
    labels: list[tuple[tuple[int, int], ...]]  # id -> canonical heads
    cfg: MonkeyConfig

    # 搜索树缓存（可选，从 precompute 加载）
    _search_tree: dict[tuple, tuple[int, int]] | None = None  # (state_key) -> (best_action, best_value)

    @staticmethod
    def from_layouts(layouts: list[dict[str, Any]], cfg: MonkeyConfig | None = None) -> "MonkeyAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        if cfg is None:
            cfg = MonkeyConfig()
        return MonkeyAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg)

    @staticmethod
    def from_precomputed(precomputed_path: str | Any, layouts: list[dict[str, Any]] | None = None) -> "MonkeyAgent":
        """从预计算的搜索树加载 agent"""
        import pickle
        from pathlib import Path
        
        path = Path(precomputed_path) if isinstance(precomputed_path, str) else precomputed_path
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        if layouts is None:
            layouts = load_layouts(None)
        
        outcomes, label_ids, labels = build_outcome_table(layouts)
        cfg = data.get("config", MonkeyConfig())
        search_tree = data.get("search_tree", None)
        
        agent = MonkeyAgent(
            outcomes=outcomes,
            label_ids=label_ids,
            labels=labels,
            cfg=cfg,
            _search_tree=search_tree,
        )
        
        return agent

    def _state_key(self, cand_idx: np.ndarray, unshot: np.ndarray, heads_hit: int) -> tuple:
        """生成状态的唯一标识（用于缓存）"""
        return (tuple(sorted(cand_idx.tolist())), tuple(unshot.tolist()), heads_hit)

    def _is_terminal(self, heads_hit: int, cand_idx: np.ndarray) -> bool:
        """判断是否达到终止状态"""
        return heads_hit >= 3 or cand_idx.size == 0

    def _terminal_steps(self, cand_idx: np.ndarray, unshot: np.ndarray, heads_hit: int) -> int:
        """
        计算终止状态下还需要的步数。
        如果已经击中 3 个机头，返回 0。
        否则，返回剩余未击中的机头数量。
        """
        if heads_hit >= 3:
            return 0
        
        if cand_idx.size == 0:
            # 不应该发生，但如果发生了，返回一个很大的数
            return 1000
        
        # 找出所有可能的机头位置
        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size == 1:
            # 只剩一种机头配置，直接数剩余未击中的机头
            heads = self.labels[int(possible_labels[0])]
            remaining = 0
            for hx, hy in heads:
                a = int(hx) * GRID_SIZE + int(hy)
                if unshot[a]:
                    remaining += 1
            return remaining
        
        # 多种可能的机头配置，返回 0（需要继续搜索）
        return 0

    def _minimax(
        self,
        cand_idx: np.ndarray,
        unshot: np.ndarray,
        heads_hit: int,
        depth: int,
        alpha: float,
        beta: float,
        is_min_player: bool,
    ) -> tuple[int | None, int]:
        """
        Minimax with alpha-beta pruning.
        搜索到终止状态（所有 3 个机头都被击中）。
        
        返回：
        - best_action: 最佳动作（如果是 None 表示终止状态）
        - best_value: 最佳值（对于 min 玩家，表示最坏情况下的剩余步数）
        """
        # 检查终止状态
        if self._is_terminal(heads_hit, cand_idx):
            return None, self._terminal_steps(cand_idx, unshot, heads_hit)

        # 检查是否只剩一种机头配置
        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size == 1:
            heads = self.labels[int(possible_labels[0])]
            remaining = 0
            for hx, hy in heads:
                a = int(hx) * GRID_SIZE + int(hy)
                if unshot[a]:
                    remaining += 1
            return None, remaining

        if is_min_player:
            # 玩家回合：选择动作（min）
            # 计算熵减并选择 top-k
            actions, gains = _compute_entropy_gains(self.outcomes, self.label_ids, cand_idx, unshot)
            
            if not actions:
                return None, self._terminal_steps(cand_idx, unshot, heads_hit)
            
            # 根据剩余候选数量决定 top_k
            current_top_k = self.cfg.top_k
            if cand_idx.size <= self.cfg.expand_threshold:
                current_top_k = self.cfg.expanded_top_k
            
            top_actions = actions[:min(current_top_k, len(actions))]
            
            best_action = top_actions[0]
            best_value = float('inf')
            
            for action in top_actions:
                # 对这个动作，考虑对手（max）的回合
                _, max_value = self._minimax_max(
                    cand_idx, unshot, heads_hit, depth, alpha, beta, action
                )
                
                if max_value < best_value:
                    best_value = max_value
                    best_action = action
                
                # Alpha-beta 剪枝
                if best_value <= alpha:
                    break
                beta = min(beta, best_value)
            
            return best_action, int(best_value)
        
        else:
            # 不应该在这里调用（max 玩家应该通过 _minimax_max 调用）
            raise RuntimeError("Should not call _minimax with is_min_player=False")

    def _minimax_max(
        self,
        cand_idx: np.ndarray,
        unshot: np.ndarray,
        heads_hit: int,
        depth: int,
        alpha: float,
        beta: float,
        action: int,
    ) -> tuple[int, int]:
        """
        Max 玩家的回合：选择结果（max）
        
        返回：
        - best_outcome: 最佳结果（0/1/2）
        - best_value: 最佳值（最大的剩余步数）
        """
        # 计算这个动作的三种可能结果的熵减
        gains, outcome_orders = _compute_entropy_gains_for_action(
            self.outcomes, self.label_ids, cand_idx, action
        )
        
        best_outcome = outcome_orders[0]
        best_value = float('-inf')
        
        for outcome_val in outcome_orders:
            # 检查这个结果是否可能
            if gains[outcome_val] == float('inf'):
                continue
            
            # 过滤候选布局
            col = self.outcomes[cand_idx, int(action)]
            new_cand_idx = cand_idx[col == outcome_val]
            
            if new_cand_idx.size == 0:
                continue
            
            # 更新状态
            new_unshot = unshot.copy()
            new_unshot[int(action)] = False
            new_heads_hit = heads_hit + (1 if outcome_val == 2 else 0)
            
            # 递归调用 min 玩家
            _, min_value = self._minimax(
                new_cand_idx, new_unshot, new_heads_hit, depth + 1, alpha, beta, is_min_player=True
            )
            
            # 加上这一步
            total_value = 1 + min_value
            
            if total_value > best_value:
                best_value = total_value
                best_outcome = outcome_val
            
            # Alpha-beta 剪枝
            if best_value >= beta:
                break
            alpha = max(alpha, best_value)
        
        return best_outcome, int(best_value)

    def choose_action(
        self,
        cand_idx: np.ndarray,
        unshot_actions: np.ndarray,
        heads_hit: int,
    ) -> int:
        """
        选择最佳动作。
        使用 minimax + alpha-beta 剪枝，搜索到终止状态。
        如果有预计算的搜索树，优先使用缓存。
        
        参数：
        - cand_idx: 候选布局索引
        - unshot_actions: 未打击的动作掩码
        - heads_hit: 已击中的机头数量
        
        返回：
        - action: 最佳动作（0-99）
        """
        # 如果只剩一种机头配置，直接打击剩余的机头
        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size == 1:
            heads = self.labels[int(possible_labels[0])]
            for hx, hy in heads:
                a = int(hx) * GRID_SIZE + int(hy)
                if unshot_actions[a]:
                    return a
        
        # 如果有预计算的搜索树，尝试从缓存中查找
        if self._search_tree is not None:
            state_key = self._state_key(cand_idx, unshot_actions, heads_hit)
            if state_key in self._search_tree:
                cached = self._search_tree[state_key]
                return cached[0]
        
        # 使用 minimax + alpha-beta 剪枝，搜索到终止状态
        best_action, best_value = self._minimax(
            cand_idx,
            unshot_actions,
            heads_hit,
            depth=0,
            alpha=self.cfg.initial_alpha,
            beta=self.cfg.initial_beta,
            is_min_player=True,
        )
        
        if best_action is None:
            # 不应该发生，但如果发生了，使用贪心策略
            actions, _ = _compute_entropy_gains(self.outcomes, self.label_ids, cand_idx, unshot_actions)
            if actions:
                return actions[0]
            # 最后的备选：返回第一个未打击的格子
            return int(np.flatnonzero(unshot_actions)[0])
        
        return best_action

    def play_one(self, env: BombPlanesEnv, *, layout: dict[str, Any], max_steps: int = 500) -> int:
        """
        玩一局游戏。
        
        返回：
        - steps: 完成游戏所需的步数
        """
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
            # 如果只剩一种机头配置，直接打击剩余的机头
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

            # 选择动作
            a = self.choose_action(cand_idx, unshot, heads_hit)
            x, y = divmod(int(a), GRID_SIZE)
            unshot[int(a)] = False

            result = shoot_xy(x, y)
            obs_v = 2 if result == GridState.HEAD else (1 if result == GridState.BODY else 0)

            # 过滤候选布局
            col = self.outcomes[cand_idx, int(a)]
            cand_idx = cand_idx[col == obs_v]

            if cand_idx.size == 0:
                break

        return steps

