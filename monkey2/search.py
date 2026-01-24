from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import GRID_SIZE
from monkey2.monkey_config import MonkeyConfig
from monkey2.state_hash import canonical_shots_tuple, state_hash_hex
from monkey2.symmetry import SymmetryGroup


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64, copy=False) / total
    return float(-(p * np.log2(p)).sum())


def _base_entropy(label_ids: np.ndarray, cand_idx: np.ndarray) -> tuple[float, np.ndarray, int]:
    """
    Returns (H(Y), inv, m).
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
    Returns (gain, head_prob).
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
    base_branch: float = 1.0  # will be overwritten to (topk)
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
        Update total_estimate based on average leaf depth:
        total ≈ (topk)^(avg_depth/2) (since Max doesn't branch)
        """
        if not self.enabled:
            return
        d = int(depth)
        if d < 0:
            return
        self.leaf_depth_sum += d
        self.leaf_count += 1
        avg = (self.leaf_depth_sum / self.leaf_count) if self.leaf_count else float(self.depth_hint)
        if not math.isfinite(avg) or avg <= 0:
            avg = float(self.depth_hint)
        # Depth is total steps (Min + Max). Min makes decisions every 2 steps.
        # But here Max is deterministic (branch factor 1).
        # So we have TopK branches every 2 levels.
        # Estimate: TopK ^ (depth/2)
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
    # Value is now (avg_steps, layout_count)
    value_cache: dict[str, tuple[float, int]]
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
    Calculate IG, sort, and pick Top K.
    Handles symmetry deduplication for equal gains.
    """
    base_h, inv64, m = _base_entropy(ctx.label_ids, cand_idx)
    features = np.flatnonzero(unshot)
    if features.size == 0:
        return []

    scored: list[tuple[float, float, int]] = []
    for a in features:
        gain, head_prob = information_gain_for_action(
            ctx.outcomes, ctx.label_ids, cand_idx, int(a), inv64=inv64, m=m, base_entropy=base_h
        )
        scored.append((float(gain), float(head_prob), int(a)))

    # Sort: gain desc, head_prob desc, action_id asc (for stability)
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

    # Determine K
    topk = int(ctx.cfg.top_k)
    if cand_idx.size <= int(ctx.cfg.small_candidates_threshold):
        topk = max(topk, int(ctx.cfg.top_k_when_small_candidates))

    # Optimization: If TopK=1, we don't need symmetry logic or loop.
    # Just take the best one.
    if topk == 1:
        if not scored:
            return []
        return [scored[0][2]]

    if not ctx.cfg.symmetry_enabled:
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
            # Already have a symmetric equivalent with this gain
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
    depth: int = 0,
) -> tuple[float, int]:
    """
    Min Node. 
    Returns (avg_steps, layout_count).
    Selects action with minimum avg_steps.
    """
    ctx.progress.tick(1)
    
    n_layouts = int(cand_idx.size)

    # Win condition: all 3 heads hit
    heads_hit = sum(1 for v in shots.values() if int(v) == 2)
    if heads_hit >= 3:
        ctx.progress.record_leaf(depth)
        return 0.0, n_layouts

    # Only one label left
    possible_labels = np.unique(ctx.label_ids[cand_idx])
    if possible_labels.size == 1:
        ctx.progress.record_leaf(depth)
        remain = _remaining_heads_when_label_fixed(ctx, int(possible_labels[0]), unshot)
        return float(remain), n_layouts

    # Memoization
    h = state_hash_hex(shots)
    cached = ctx.value_cache.get(h)
    if cached is not None:
        return cached

    actions = _pick_topk_actions(ctx, cand_idx, unshot, shots)
    if not actions:
        # Should not happen
        res = (1000.0, n_layouts)
        ctx.value_cache[h] = res
        return res

    best_avg = 1e9
    best_count = n_layouts
    best_a = int(actions[0])

    # Search Top K
    for a in actions:
        # Recurse: Max node returns (avg, count) for the sub-tree
        # where avg is the average REMAINING steps.
        # So for this action, steps = 1 + sub_avg
        sub_avg, _ = _solve_max_after_action(ctx, cand_idx, unshot, shots, int(a), depth=depth)
        
        val = 1.0 + sub_avg
        
        if val < best_avg:
            best_avg = val
            best_a = int(a)

    res = (best_avg, best_count)
    ctx.value_cache[h] = res
    ctx.best_action_cache[h] = int(best_a)
    return res


def _solve_max_after_action(
    ctx: SearchContext,
    cand_idx: np.ndarray,
    unshot: np.ndarray,
    shots: dict[int, int],
    a: int,
    *,
    depth: int,
) -> tuple[float, int]:
    """
    Max Node (Environment).
    Only recurses on the worst branch (Highest Entropy / Largest Count).
    Returns (avg_remaining_steps, total_layouts).
    """
    ctx.progress.tick(1)
    total_layouts = int(cand_idx.size)

    if not bool(unshot[int(a)]):
        return 1000.0, total_layouts

    unshot2 = unshot.copy()
    unshot2[int(a)] = False

    branches: list[tuple[float, int, np.ndarray]] = []
    
    # Check all possible outcomes
    for v in (0, 1, 2):
        col = ctx.outcomes[cand_idx, int(a)]
        sub = cand_idx[col == int(v)]
        if sub.size == 0:
            continue
            
        # We use Entropy to pick the "Worst" branch as per original logic.
        # (Usually correlates with largest count)
        _, inv = np.unique(ctx.label_ids[sub], return_inverse=True)
        h = _entropy_from_counts(np.bincount(inv, minlength=int(inv.max() + 1) if inv.size else 0).astype(np.float64, copy=False))
        branches.append((float(h), int(v), sub))

    if not branches:
        return 0.0, total_layouts

    # Sort by entropy DESCENDING (Highest entropy = Worst case)
    branches.sort(key=lambda t: -t[0])

    # 1. Recurse on the WORST branch
    _, v_worst, sub_worst = branches[0]
    n_worst = int(sub_worst.size)
    
    shots2 = dict(shots)
    shots2[int(a)] = int(v_worst)
    
    avg_worst, _ = solve_minimax_and_record_best_action(ctx, sub_worst, unshot2, shots2, depth=depth + 2)
    
    # 2. Calculate Weighted Average
    # Total Avg = (N_worst * Avg_worst + Sum(N_other * Avg_other)) / N_total
    # Assumption: Avg_other = 0 (optimistic estimate for pruned branches)
    
    weighted_sum = float(n_worst) * avg_worst
    
    # If there are other branches, we assume they take 0 extra steps (optimistic)
    # or 1 extra step? Let's stick to 0 for "remaining steps".
    # (Since we stop searching, we assume they are resolved quickly or we don't care)
    
    final_avg = weighted_sum / total_layouts if total_layouts > 0 else 0.0
    
    return final_avg, total_layouts


def make_progress(cfg: MonkeyConfig, *, enabled: bool | None = None) -> Progress:
    en = cfg.progress_enabled if enabled is None else bool(enabled)
    topk = max(int(cfg.top_k), 1)
    d = max(int(cfg.progress_depth_hint), 1)
    base_branch = float(topk) 
    # Estimate: base_branch ^ (depth/2)
    total = float((base_branch) ** (d / 2))
    now = time.time()
    return Progress(enabled=en, total_estimate=total, start_t=now, last_print_t=now, base_branch=base_branch, depth_hint=d)
