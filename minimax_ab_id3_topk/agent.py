from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import GRID_SIZE, GridState
from environment import BombPlanesEnv, build_outcome_table
from minimax_ab_id3_topk.config import MiniMaxABID3TopKConfig


class _Progress:
    def __init__(self, *, cfg: MiniMaxABID3TopKConfig) -> None:
        self.cfg = cfg
        self.t0 = time.time()
        self.last_print = self.t0
        self.last_visited = 0

    def update(
        self,
        *,
        counters: dict[str, Any],
        tt_size: int,
        depth_now: int,
    ) -> None:
        if not bool(self.cfg.progress_enabled):
            return
        now = time.time()
        if now - self.last_print < float(self.cfg.progress_every_sec):
            return

        visited = int(counters.get("visited", 0))
        tt_hits = int(counters.get("tt_hits", 0))
        depth_sum = float(counters.get("depth_sum", 0.0))
        depth_cnt = max(1, int(counters.get("depth_cnt", 0)))
        avg_steps = (depth_sum / depth_cnt) / 2.0  # "当前步数平均值"（平均 step）

        elapsed = now - self.t0
        dv = visited - self.last_visited
        dt = max(1e-9, now - self.last_print)
        nps = dv / dt

        # Dynamic rough estimate of total nodes (user-requested):
        # est_total_nodes ≈ (breadth=top_k*3)^(depth=avg_steps)
        breadth = float(max(1, int(self.cfg.top_k) * 3))
        depth_est = max(avg_steps, float(depth_now) / 2.0)
        est_total = int(min(1_000_000_000, max(1.0, breadth ** depth_est)))

        frac = min(1.0, visited / max(1, est_total))
        w = 24  # fixed width; config removed
        filled = int(w * frac)
        bar = "=" * filled + " " * (w - filled)
        msg = (
            f"\r[ab_id3k {bar}] "
            f"visited {visited} /~{est_total}  tt {tt_size} hits {tt_hits}  "
            f"avg_depth {avg_steps:5.2f}  cur_depth {float(depth_now)/2.0:5.2f}  "
            f"{nps:7.0f} n/s  {elapsed:6.1f}s"
        )

        print(msg, end="", flush=True)
        self.last_print = now
        self.last_visited = visited


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def _pack_unshot(unshot: np.ndarray) -> bytes:
    # 100 bool -> 13 bytes (packed bits)
    return np.packbits(unshot.astype(np.uint8), bitorder="little").tobytes()


def _hash_cand_idx(cand_idx: np.ndarray) -> bytes:
    # 8-byte digest, extremely low collision risk in practice; keyed with size as well.
    return blake2b(cand_idx.tobytes(), digest_size=8).digest()


def _id3_topk_actions(
    outcomes: np.ndarray,
    label_ids: np.ndarray,
    cand_idx: np.ndarray,
    unshot: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """
    Compute ID3 information-gain for each unshot action under current candidates, return top-k action ids.
    Tie-break: higher head_prob, then smaller action id.
    """
    feats = np.flatnonzero(unshot)
    if feats.size == 0:
        return feats
    if feats.size <= k:
        return feats.astype(np.int32, copy=False)

    y = label_ids[cand_idx]
    uniq, inv = np.unique(y, return_inverse=True)
    m = int(uniq.size)
    # When m==1, IG is 0 for all; we fall back to head_prob.
    base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m)) if m > 0 else 0.0

    inv64 = inv.astype(np.int64, copy=False)
    gains = np.empty((feats.size,), dtype=np.float64)
    head_probs = np.empty((feats.size,), dtype=np.float64)

    for i, a in enumerate(feats):
        col = outcomes[cand_idx, int(a)].astype(np.int64, copy=False)  # 0/1/2
        combo = col * m + inv64
        cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
        outcome_counts = cont.sum(axis=1)
        n = int(outcome_counts.sum())
        if n <= 0:
            gains[i] = -1e100
            head_probs[i] = -1.0
            continue

        if m <= 1:
            cond_entropy = 0.0
        else:
            cond_entropy = 0.0
            for v in range(3):
                nv = int(outcome_counts[v])
                if nv == 0:
                    continue
                cond_entropy += (nv / n) * _entropy_from_counts(cont[v])
        gains[i] = float(base_entropy - cond_entropy)
        head_probs[i] = float(outcome_counts[2] / n)

    k = max(1, int(k))
    top_idx = np.argpartition(-gains, kth=k - 1)[:k]
    top_actions = feats[top_idx].astype(np.int32, copy=False)
    top_gains = gains[top_idx]
    top_hp = head_probs[top_idx]

    # sort by (-gain, -head_prob, action_id)
    order = np.lexsort((top_actions, -top_hp, -top_gains))
    return top_actions[order]


def _state_cache_key(*, heads_hit: int, cand_idx: np.ndarray, unshot: np.ndarray, top_k: int) -> str:
    """
    Cache key for a belief-state + branching config.
    """
    h = blake2b(digest_size=16)
    h.update(int(heads_hit).to_bytes(1, "little", signed=False))
    h.update(int(top_k).to_bytes(2, "little", signed=False))
    h.update(_hash_cand_idx(cand_idx))
    h.update(_pack_unshot(unshot))
    return h.hexdigest()


INF = 10**9


@dataclass
class MiniMaxABID3TopKAgent:
    """
    MiniMax + alpha-beta pruning, where branching on MIN actions is limited by ID3 top-k.

    Game model (adversarial observation):
    - State is a belief set over layouts (cand_idx) plus unshot cells plus heads_hit.
    - MIN chooses an unshot cell to attack (we only expand ID3 top-k).
    - MAX chooses the outcome among {MISS,BODY,HEAD} that is *still possible* under the belief,
      to maximize remaining steps to termination.
    Terminal conditions (value returns "remaining steps"):
    - If heads_hit==3: 0.
    - If belief collapses to a single head-configuration label: remaining_heads (count of unshot heads).
    """

    outcomes: np.ndarray  # (N,100) uint8 in {0,1,2}
    label_ids: np.ndarray  # (N,) int32
    labels: list[tuple[tuple[int, int], ...]]  # id -> canonical heads (3 tuples)
    cfg: MiniMaxABID3TopKConfig = MiniMaxABID3TopKConfig()
    _policy_index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _cache_loaded: bool = field(default=False, init=False, repr=False)
    _cache_nodes: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def _cache_file_path(self) -> Path:
        # One unified cache file per top_k (so you don't get "a bunch of files").
        return Path(self.cfg.cache_dir) / f"policy_cache_topk{int(self.cfg.top_k)}.json"

    def _load_cache_once(self) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        if not bool(self.cfg.cache_enabled):
            return
        p = self._cache_file_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            nodes = data.get("nodes")
            if isinstance(nodes, dict):
                # nodes: state_key -> MIN node dict
                self._cache_nodes = {str(k): v for k, v in nodes.items() if isinstance(v, dict)}
                # also seed policy_index for fast lookup
                for sk, n in self._cache_nodes.items():
                    if "best_action" in n:
                        self._policy_index[sk] = n
        except Exception:
            # ignore corrupted cache
            self._cache_nodes = {}

    def _save_cache(self) -> None:
        if not bool(self.cfg.cache_enabled):
            return
        try:
            Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)
            p = self._cache_file_path()
            tmp = p.with_suffix(".tmp")
            payload = {
                "version": 1,
                "top_k": int(self.cfg.top_k),
                "tree_log_depth": int(self.cfg.tree_log_depth),
                "nodes": self._cache_nodes,
                "saved_at_unix": float(time.time()),
            }
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            pass

    def _index_policy_tree(self, node: dict[str, Any] | None) -> None:
        """
        Index all MIN nodes in a cached/constructed policy tree by their state_key.
        """
        if not isinstance(node, dict):
            return

        stack: list[dict[str, Any]] = [node]
        while stack:
            cur = stack.pop()
            if not isinstance(cur, dict):
                continue
            if cur.get("kind") == "MIN":
                sk = cur.get("state_key")
                if isinstance(sk, str) and ("best_action" in cur):
                    self._policy_index[sk] = cur
                    self._cache_nodes[sk] = cur
                # traverse outcomes
                outs = cur.get("outcomes")
                if isinstance(outs, list):
                    for o in outs:
                        if isinstance(o, dict):
                            ch = o.get("child")
                            if isinstance(ch, dict):
                                stack.append(ch)

    @staticmethod
    def from_layouts(
        layouts: list[dict[str, Any]],
        *,
        top_k: int = 8,
        cfg: MiniMaxABID3TopKConfig | None = None,
    ) -> "MiniMaxABID3TopKAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        if cfg is None:
            cfg = MiniMaxABID3TopKConfig(top_k=int(top_k))
        else:
            cfg = MiniMaxABID3TopKConfig(
                top_k=int(top_k),
                progress_enabled=cfg.progress_enabled,
                progress_every_sec=cfg.progress_every_sec,
                tree_log_depth=cfg.tree_log_depth,
                cache_enabled=cfg.cache_enabled,
                cache_dir=cfg.cache_dir,
            )
        return MiniMaxABID3TopKAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg)

    def _terminal_remaining_heads(self, cand_idx: np.ndarray, unshot: np.ndarray) -> int | None:
        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size != 1:
            return None
        heads = self.labels[int(possible_labels[0])]
        rem = 0
        for hx, hy in heads:
            a = int(hx) * GRID_SIZE + int(hy)
            if bool(unshot[a]):
                rem += 1
        return int(rem)

    def _solve_min_state(
        self,
        cand_idx: np.ndarray,
        unshot: np.ndarray,
        heads_hit: int,
        alpha: int,
        beta: int,
        tt: dict[tuple[int, int, bytes, bytes], int],
        counters: dict[str, Any],
        *,
        depth_left: int,
        depth_now: int,
        progress: _Progress | None,
    ) -> tuple[int, dict[str, Any] | None]:
        """
        Solve a MIN belief-state exactly (within ID3 top-k branching) and optionally build a *policy tree*
        of the first `depth_left` plies:
        - MIN node stores one best action
        - then has up to 3 OBS children (0/1/2) (only those that are possible)
        - each OBS child leads to the next MIN node (unique best action), recursively.
        """
        # Terminal checks:
        # - All 3 heads hit
        # - Belief collapses to exactly one head-configuration label: remaining steps = remaining head cells
        state_key = _state_cache_key(heads_hit=int(heads_hit), cand_idx=cand_idx, unshot=unshot, top_k=int(self.cfg.top_k))
        if heads_hit >= 3:
            return (
                0,
                {"kind": "MIN", "state_key": state_key, "meta": {"cand": int(cand_idx.size), "heads_hit": int(heads_hit)}, "value": 0}
                if depth_left > 0
                else None,
            )
        if cand_idx.size == 0 or (not unshot.any()):
            return (
                INF,
                {"kind": "MIN", "state_key": state_key, "meta": {"cand": int(cand_idx.size), "heads_hit": int(heads_hit)}, "value": INF}
                if depth_left > 0
                else None,
            )

        rem = self._terminal_remaining_heads(cand_idx, unshot)
        if rem is not None:
            node = {
                "kind": "MIN",
                "state_key": state_key,
                "terminal": True,
                "meta": {"cand": int(cand_idx.size), "heads_hit": int(heads_hit)},
                "value": int(rem),
            }
            return int(rem), node if depth_left > 0 else None

        counters["visited"] += 1
        counters["depth_sum"] = float(counters.get("depth_sum", 0.0)) + float(depth_now)
        counters["depth_cnt"] = int(counters.get("depth_cnt", 0)) + 1
        if progress is not None:
            progress.update(counters=counters, tt_size=len(tt), depth_now=int(depth_now))

        key = (int(heads_hit), int(cand_idx.size), _hash_cand_idx(cand_idx), _pack_unshot(unshot))
        cached = tt.get(key)
        if cached is not None and depth_left <= 0:
            counters["tt_hits"] += 1
            return int(cached), None

        actions = _id3_topk_actions(self.outcomes, self.label_ids, cand_idx, unshot, k=int(self.cfg.top_k))
        if actions.size == 0:
            return (
                INF,
                {"kind": "MIN", "state_key": state_key, "meta": {"cand": int(cand_idx.size), "heads_hit": int(heads_hit)}, "value": INF}
                if depth_left > 0
                else None,
            )

        best_v = INF
        best_a = int(actions[0])

        # alpha/beta bounds for child state values: if child returns t, total is 1+t.
        child_alpha = alpha - 1
        child_beta = beta - 1

        for a in actions:
            if not bool(unshot[int(a)]):
                continue
            unshot2 = unshot.copy()
            unshot2[int(a)] = False
            col = self.outcomes[cand_idx, int(a)]

            # MAX chooses worst outcome among possible {0/1/2}
            worst_child = -1

            # Heuristic ordering for alpha-beta effectiveness:
            # Explore MAX outcomes likely to be "worse" first (more remaining ambiguity).
            # We use unique-label count as a proxy for remaining steps.
            outcome_candidates: list[tuple[int, int, int]] = []  # (proxy, v, cand2_size)
            for v in (0, 1, 2):
                mask = col == v
                if not bool(mask.any()):
                    continue
                cand2 = cand_idx[mask]
                uniq_labels = int(np.unique(self.label_ids[cand2]).size)
                outcome_candidates.append((uniq_labels, int(v), int(cand2.size)))
            # sort by proxy desc, then size desc, then v (stable)
            outcome_candidates.sort(key=lambda t: (t[0], t[2], t[1]), reverse=True)

            for _, v, _sz in outcome_candidates:
                mask = col == v
                if not bool(mask.any()):
                    continue
                cand2 = cand_idx[mask]
                heads2 = heads_hit + (1 if v == 2 else 0)
                t_child, _child_tree = self._solve_min_state(
                    cand2,
                    unshot2,
                    heads2,
                    child_alpha,
                    child_beta,
                    tt,
                    counters,
                    depth_left=max(0, depth_left - 2),
                    depth_now=int(depth_now) + 2,
                    progress=progress,
                )
                if t_child > worst_child:
                    worst_child = int(t_child)

                # alpha-beta pruning for the MAX aggregation
                if worst_child >= child_beta:
                    break
                if worst_child > child_alpha:
                    child_alpha = worst_child

            val = 1 + int(worst_child)
            if val < best_v:
                best_v = int(val)
                best_a = int(a)

            if best_v <= alpha:
                break
            if best_v < beta:
                beta = best_v
                child_beta = beta - 1

        tt[key] = int(best_v)
        if depth_left <= 0:
            return int(best_v), None

        # Build *policy tree* for the unique best action only (3 outcomes),
        # while preserving alpha-beta during best-action search above.
        # This matches: MIN has unique choice; then 0/1/2; then each leads to unique MIN choice.
        unshot2 = unshot.copy()
        unshot2[int(best_a)] = False
        col = self.outcomes[cand_idx, int(best_a)]
        best_outcomes: list[dict[str, Any]] = []
        for v in (0, 1, 2):
            mask = col == v
            if not bool(mask.any()):
                continue
            cand2 = cand_idx[mask]
            heads2 = heads_hit + (1 if v == 2 else 0)
            t_child, child_tree = self._solve_min_state(
                cand2,
                unshot2,
                heads2,
                -10**9,
                10**9,
                tt,
                counters,
                depth_left=max(0, depth_left - 2),
                depth_now=int(depth_now) + 2,
                progress=progress,
            )
            best_outcomes.append(
                {
                    "kind": "OBS",
                    "obs": int(v),
                    "meta": {"cand": int(cand2.size), "heads_hit": int(heads2)},
                    "value": int(1 + int(t_child)),
                    "child": child_tree,
                }
            )

        node = {
            "kind": "MIN",
            "state_key": state_key,
            "meta": {"cand": int(cand_idx.size), "heads_hit": int(heads_hit)},
            "best_action": int(best_a),
            "value": int(best_v),
            "outcomes": best_outcomes,
        }
        return int(best_v), node

    def choose_action(self, cand_idx: np.ndarray, unshot_actions: np.ndarray, heads_hit: int) -> int:
        if not unshot_actions.any():
            raise ValueError("No available actions.")

        # cache lookup
        cache_key = _state_cache_key(heads_hit=int(heads_hit), cand_idx=cand_idx, unshot=unshot_actions, top_k=int(self.cfg.top_k))
        self._load_cache_once()

        # 1) in-memory policy index (fast path)
        n = self._policy_index.get(cache_key)
        if isinstance(n, dict) and ("best_action" in n):
            return int(n["best_action"])

        # Shortcut: if only one head-set label remains, directly shoot remaining heads.
        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size == 1:
            heads = self.labels[int(possible_labels[0])]
            for hx, hy in heads:
                a = int(hx) * GRID_SIZE + int(hy)
                if bool(unshot_actions[a]):
                    return int(a)

        # Search: evaluate each candidate action (ID3 top-k) by minimax value.
        tt: dict[tuple[int, int, bytes, bytes], int] = {}
        counters: dict[str, Any] = {"visited": 0, "tt_hits": 0, "depth_sum": 0.0, "depth_cnt": 0}
        progress = _Progress(cfg=self.cfg) if bool(self.cfg.progress_enabled) else None

        v, policy_tree = self._solve_min_state(
            cand_idx=cand_idx,
            unshot=unshot_actions,
            heads_hit=int(heads_hit),
            alpha=-INF,
            beta=INF,
            tt=tt,
            counters=counters,
            depth_left=int(self.cfg.tree_log_depth),
            depth_now=0,
            progress=progress,
        )

        if bool(self.cfg.progress_enabled):
            # final line
            avg_steps = (float(counters.get("depth_sum", 0.0)) / max(1, int(counters.get("depth_cnt", 1)))) / 2.0
            print(
                f"\r[ab_id3k done] visited {int(counters['visited'])}  tt {len(tt)} hits {int(counters['tt_hits'])}  "
                f"avg_depth {avg_steps:5.2f}  value {int(v)}",
                flush=True,
            )

        best_a = int(policy_tree["best_action"]) if isinstance(policy_tree, dict) and ("best_action" in policy_tree) else int(np.flatnonzero(unshot_actions)[0])
        best_v = int(v)

        # index in-memory policy tree for reuse in subsequent steps (within saved depth)
        self._index_policy_tree(policy_tree if isinstance(policy_tree, dict) else None)
        self._save_cache()

        return int(best_a)

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
            if not unshot.any():
                break

            a = self.choose_action(cand_idx=cand_idx, unshot_actions=unshot, heads_hit=heads_hit)
            x, y = divmod(int(a), GRID_SIZE)
            unshot[int(a)] = False

            result = shoot_xy(int(x), int(y))
            obs_v = 2 if result == GridState.HEAD else (1 if result == GridState.BODY else 0)

            col = self.outcomes[cand_idx, int(a)]
            cand_idx = cand_idx[col == obs_v]
            if cand_idx.size == 0:
                break

        return steps


