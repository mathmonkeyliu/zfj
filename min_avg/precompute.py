from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import GRID_SIZE
from environment import load_layouts, build_outcome_table
from .agent import MinAvgConfig, MinAvgPlanner, MinAvgPolicy


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
    
    # 构建 Config
    cfg_args = {}
    if args.topk is not None:
        cfg_args["top_k"] = int(args.topk)
    if args.silent:
        cfg_args["progress_enabled"] = False
    
    cfg = MinAvgConfig(**cfg_args)

    print(f"Starting precompute with {len(layouts)} layouts, top_k={cfg.top_k}...")
    print(f"Note: Total nodes visited is expected to be approx 1.5x - 2x the number of layout groups (Leaves + Internal Decision Nodes).")

    planner = MinAvgPlanner(outcomes, label_ids, labels, cfg)
    init_board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    cand_idx = np.arange(outcomes.shape[0], dtype=np.int32)
    
    # 开始搜索
    # 搜索完成后，policy_map 中只包含最优路径上的 (state -> best_action)
    _, total_steps, total_count, policy_map = planner.search_state(init_board, cand_idx)
    
    avg_steps = total_steps / total_count if total_count > 0 else 0
    print(f"\nSearch complete.")
    print(f"Global average steps: {avg_steps:.4f}")
    print(f"Total nodes visited: {planner._visited_nodes}")
    print(f"Policy size (states recorded): {len(policy_map)}")
    
    policy = MinAvgPolicy(policy=policy_map, grid_size=GRID_SIZE)
    out_path = Path(args.out)
    policy.save(out_path)
    print(f"[min_avg] policy saved to: {out_path}")


if __name__ == "__main__":
    main()
