from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import GRID_SIZE
from environment import load_layouts
from .agent import MinAvgConfig, MinAvgPlanner, MinAvgPolicy
from environment import build_outcome_table


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute min_avg greedy policy (best action per state).")
    ap.add_argument("--out", type=str, default="min_avg_policy.json", help="Output policy file path.")
    ap.add_argument("--topk", type=int, default=None, help="Override top_k branching factor.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of layouts (debug).")
    ap.add_argument("--silent", action="store_true", help="Disable progress output.")
    args = ap.parse_args()

    layouts = load_layouts(None)
    if args.limit is not None:
        layouts = layouts[: args.limit]

    outcomes, label_ids, labels = build_outcome_table(layouts)
    cfg = MinAvgConfig()
    if args.topk is not None:
        cfg = MinAvgConfig(top_k=int(args.topk), progress_enabled=cfg.progress_enabled, progress_every=cfg.progress_every)
    if args.silent:
        cfg = MinAvgConfig(top_k=cfg.top_k, progress_enabled=False, progress_every=cfg.progress_every)

    planner = MinAvgPlanner(outcomes, label_ids, labels, cfg)
    init_board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    cand_idx = np.arange(outcomes.shape[0], dtype=np.int32)
    planner.search(init_board, cand_idx)

    policy = MinAvgPolicy(policy=planner.policy, grid_size=GRID_SIZE)
    out_path = Path(args.out)
    policy.save(out_path)
    print(f"[min_avg] policy saved: {out_path} | states: {len(policy.policy)}")


if __name__ == "__main__":
    main()
