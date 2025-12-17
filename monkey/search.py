from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import GRID_SIZE
from monkey.monkey_config import MonkeyConfig
from monkey.state_hash import canonical_shots_tuple, state_hash_hex
from monkey.symmetry import SymmetryGroup


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64, copy=False) / total
    return float(-(p * np.log2(p)).sum())


def _base_entropy(label_ids: np.ndarray, cand_idx: np.ndarray) -> tuple[float, np.ndarray, int]:
    """
    返回 (H(Y), inv, m)，其中 inv 是 cand_idx 对应 label 的压缩编号，m 是不同 label 数。
    """
    y = label_ids[cand_idx]
    _, inv = np.unique(y, return_inverse=True)
    m = int(inv.max() + 1) if inv.size else 0
    base_h = _entropy_from_counts(np.bincount(inv, minlength=m).astype(np.float64, copy=False))
    return float(base_h), inv.astype(np.int64, copy=False), int(m)


def information_gain_for_action(
    outcomes: np.ndarray,
    label_ids: np.ndarray,
    cand_idx: np.ndarray,
    a: int,
    *,
    inv64: np.ndarray,
    m: int,
    base_entropy: float,
) -> tuple[float, float]:
    """
    返回 (gain, head_prob)。
    head_prob 用于与 ID3 保持一致：当 gain 相同，优先 head_prob 更大者。
    """
    col = outcomes[cand_idx, int(a)].astype(np.int64, copy=False)
    combo = col * m + inv64
    cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
    outcome_counts = cont.sum(axis=1)
    n = int(outcome_counts.sum())
    if n == 0:
        return 0.0, 0.0

    cond_entropy = 0.0
    for v in range(3):
        nv = int(outcome_counts[v])
        if nv == 0:
            continue
        cond_entropy += (nv / n) * _entropy_from_counts(cont[v])

    gain = float(base_entropy - cond_entropy)
    head_prob = float(outcome_counts[2] / n)
    return gain, head_prob


@dataclass(slots=True)
class Progress:
    enabled: bool
    total_estimate: float
    start_t: float
    last_print_t: float
    visited: int = 0
    # dynamic depth estimate: average depth of leaf nodes encountered so far
    leaf_depth_sum: int = 0
    leaf_count: int = 0
    base_branch: float = 3.0  # will be overwritten to (topk*3)
    depth_hint: int = 14

    def tick(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self.visited += int(n)
        now = time.time()
        if now - self.last_print_t < 0.2:
            return
        self.last_print_t = now
        total = max(float(self.total_estimate), 1.0)
        frac = min(self.visited / total, 1.0)
        bar_w = 30
        filled = int(bar_w * frac)
        bar = "=" * filled + " " * (bar_w - filled)
        elapsed = now - self.start_t
        rate = self.visited / elapsed if elapsed > 0 else 0.0
        eta = (total - self.visited) / rate if rate > 0 else float("inf")
        eta_str = f"{eta:6.1f}s" if math.isfinite(eta) else "  inf s"
        sys.stdout.write(
            f"\r[{bar}] {frac*100:6.2f}%  nodes {self.visited:9d}/{int(total):9d}  elapsed {elapsed:6.1f}s  eta {eta_str}"
        )
        sys.stdout.flush()

    def done(self) -> None:
        if not self.enabled:
            return
        self.tick(0)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def record_leaf(self, depth: int) -> None:
        """
        用叶子平均深度动态更新 total_estimate：
        total ≈ (topk*3)^(avg_depth/2)
        """
        if not self.enabled:
            return
        d = int(depth)
        if d < 0:
            return
        self.leaf_depth_sum += d
        self.leaf_count += 1
        avg = (self.leaf_depth_sum / self.leaf_count) if self.leaf_count else float(self.depth_hint)
        # 如果叶子还没出现，就用 hint
        if not math.isfinite(avg) or avg <= 0:
            avg = float(self.depth_hint)
        self.total_estimate = float((self.base_branch) ** (avg / 2.0))


@dataclass(slots=True)
class SearchContext:
    outcomes: np.ndarray
    label_ids: np.ndarray
    labels: list[tuple[tuple[int, int], ...]]
    cfg: MonkeyConfig
    sym: SymmetryGroup
    progress: Progress

    # transposition tables (min states)
    value_cache: dict[str, int]
    best_action_cache: dict[str, int]  # state_hash -> action_id


def _remaining_heads_when_label_fixed(ctx: SearchContext, label_id: int, unshot: np.ndarray) -> int:
    heads = ctx.labels[int(label_id)]
    remain = 0
    for hx, hy in heads:
        a = int(hx) * GRID_SIZE + int(hy)
        if bool(unshot[int(a)]):
            remain += 1
    return int(remain)


def _pick_topk_actions(
    ctx: SearchContext,
    cand_idx: np.ndarray,
    unshot: np.ndarray,
    shots: dict[int, int],
) -> list[int]:
    """
    计算所有未点击格子的 IG，并按 IG 降序排序后取 TopK。
    若 IG 相同（round 后同一桶），做“对称去重”（只保留第一个）。
    """
    base_h, inv64, m = _base_entropy(ctx.label_ids, cand_idx)
    features = np.flatnonzero(unshot)
    if features.size == 0:
        return []

    # TopK=1 时，严格复刻 ID3 的扫描逻辑（含 np.isclose tie-break），保证与 id3 一致
    if int(ctx.cfg.top_k) == 1:
        best_gain = -1.0
        best_a = int(features[0])
        best_head_prob = -1.0
        for a in features:
            gain, head_prob = information_gain_for_action(
                ctx.outcomes, ctx.label_ids, cand_idx, int(a), inv64=inv64, m=m, base_entropy=base_h
            )
            if (gain > best_gain) or (np.isclose(gain, best_gain) and head_prob > best_head_prob):
                best_gain = float(gain)
                best_a = int(a)
                best_head_prob = float(head_prob)
        return [int(best_a)]

    scored: list[tuple[float, float, int]] = []
    for a in features:
        gain, head_prob = information_gain_for_action(
            ctx.outcomes, ctx.label_ids, cand_idx, int(a), inv64=inv64, m=m, base_entropy=base_h
        )
        scored.append((float(gain), float(head_prob), int(a)))

    # sort: gain desc, head_prob desc, action_id asc
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

    topk = int(ctx.cfg.top_k)
    if cand_idx.size <= int(ctx.cfg.small_candidates_threshold):
        topk = max(topk, int(ctx.cfg.top_k_when_small_candidates))

    if (not ctx.cfg.symmetry_enabled) or topk <= 1:
        return [a for _, _, a in scored[:topk]]

    stabilizer = (
        ctx.sym.stabilizer_transforms(shots) if ctx.cfg.symmetry_consider_state else list(range(len(ctx.sym.maps)))
    )

    selected: list[int] = []
    seen_by_gain: dict[float, set[int]] = {}
    nd = int(ctx.cfg.symmetry_gain_round_ndigits)

    for gain, _hp, a in scored:
        if len(selected) >= topk:
            break
        gk = round(float(gain), nd)
        rep = ctx.sym.canonical_action_under_stabilizer(int(a), stabilizer)
        s = seen_by_gain.get(gk)
        if s is None:
            s = set()
            seen_by_gain[gk] = s
        if rep in s:
            continue
        s.add(rep)
        selected.append(int(a))

    return selected


def solve_minimax_and_record_best_action(
    ctx: SearchContext,
    cand_idx: np.ndarray,
    unshot: np.ndarray,
    shots: dict[int, int],
    *,
    alpha: int = -10**9,
    beta: int = 10**9,
    depth: int = 0,
) -> int:
    """
    返回当前 MIN（玩家）节点在最坏反馈下的最少剩余步数。
    同时会在 ctx.best_action_cache 里记录该状态最优下一步动作（action_id）。
    """
    ctx.progress.tick(1)

    # 终局：已击中 3 个机头
    heads_hit = sum(1 for v in shots.values() if int(v) == 2)
    if heads_hit >= 3:
        ctx.progress.record_leaf(depth)
        return 0

    # 终止规则：只剩一个机头三元组 label -> 直接点剩余机头（确定最优）
    possible_labels = np.unique(ctx.label_ids[cand_idx])
    if possible_labels.size == 1:
        ctx.progress.record_leaf(depth)
        return _remaining_heads_when_label_fixed(ctx, int(possible_labels[0]), unshot)

    # memo
    h = state_hash_hex(shots)
    cached = ctx.value_cache.get(h)
    if cached is not None:
        return int(cached)

    actions = _pick_topk_actions(ctx, cand_idx, unshot, shots)
    if not actions:
        # 没有动作可选：视为无法继续（理论不会发生），返回一个大值
        ctx.value_cache[h] = 10**6
        return 10**6

    best_val = 10**9
    best_a = int(actions[0])

    for a in actions:
        worst_after = _solve_max_after_action(ctx, cand_idx, unshot, shots, int(a), alpha=alpha, beta=beta, depth=depth)
        v = 1 + int(worst_after)
        if v < best_val:
            best_val = int(v)
            best_a = int(a)
        if ctx.cfg.alpha_beta:
            beta = min(beta, best_val)
            if beta <= alpha:
                break

    ctx.value_cache[h] = int(best_val)
    ctx.best_action_cache[h] = int(best_a)
    return int(best_val)


def _solve_max_after_action(
    ctx: SearchContext,
    cand_idx: np.ndarray,
    unshot: np.ndarray,
    shots: dict[int, int],
    a: int,
    *,
    alpha: int,
    beta: int,
    depth: int,
) -> int:
    """
    MAX 节点：对玩家选的格子 a 给出反馈（0/1/2），目标使步数最大。
    分支为空（不对应任何布局）直接舍弃。
    分支搜索顺序：剩余信息熵最大者优先（利于 αβ 剪枝）。
    """
    ctx.progress.tick(1)

    if not bool(unshot[int(a)]):
        return 10**6

    unshot2 = unshot.copy()
    unshot2[int(a)] = False

    branches: list[tuple[float, int, np.ndarray]] = []
    for v in (0, 1, 2):
        col = ctx.outcomes[cand_idx, int(a)]
        sub = cand_idx[col == int(v)]
        if sub.size == 0:
            continue
        # branch entropy (larger first)
        _, inv = np.unique(ctx.label_ids[sub], return_inverse=True)
        h = _entropy_from_counts(np.bincount(inv, minlength=int(inv.max() + 1) if inv.size else 0).astype(np.float64, copy=False))
        branches.append((float(h), int(v), sub))

    if not branches:
        return -10**6

    branches.sort(key=lambda t: -t[0])

    best = -10**9
    for _h, v, sub in branches:
        shots2 = dict(shots)
        shots2[int(a)] = int(v)
        val = solve_minimax_and_record_best_action(ctx, sub, unshot2, shots2, alpha=alpha, beta=beta, depth=depth + 2)
        best = max(best, int(val))
        if ctx.cfg.alpha_beta:
            alpha = max(alpha, best)
            if beta <= alpha:
                break

    return int(best)


def make_progress(cfg: MonkeyConfig, *, enabled: bool | None = None) -> Progress:
    en = cfg.progress_enabled if enabled is None else bool(enabled)
    # total estimate: (topk*3)^(depth/2)
    topk = max(int(cfg.top_k), 1)
    d = max(int(cfg.progress_depth_hint), 1)
    base_branch = float(topk * 3)
    total = float((base_branch) ** (d / 2))
    now = time.time()
    return Progress(enabled=en, total_estimate=total, start_t=now, last_print_t=now, base_branch=base_branch, depth_hint=d)


