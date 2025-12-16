from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any
import sys
import time

import numpy as np

from environment import BombPlanesEnv, load_layouts
from id3 import ID3Agent
from elim import ElimAgent
from mcts import MCTSAgent, MCTSConfig
from monkey import MonkeyAgent, MonkeyConfig

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a method across all layouts (metric: steps to hit all 3 heads).")
    ap.add_argument("--method", choices=["id3", "elim", "mcts", "monkey"], default="id3")
    ap.add_argument("--layouts-file", default=None, help="Path to layouts.jsonl (defaults to config.LAYOUT_FILE).")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of layouts to evaluate.")
    ap.add_argument("--out", default=None, help="Output png path.")
    ap.add_argument("--mcts-sims", type=int, default=None, help="MCTS: simulations per move (overrides mcts/config.py default).")
    ap.add_argument("--mcts-depth", type=int, default=None, help="MCTS: max search depth (overrides mcts/config.py default).")
    # monkey is configured via monkey/config.py (edit that file to tune).
    args = ap.parse_args()

    # Candidate universe for the online algorithm: all layouts from file.
    all_layouts = load_layouts(args.layouts_file)

    # Evaluation set: default = all layouts; or a prefix via --limit for quick tests.
    layouts = all_layouts
    if args.limit is not None:
        layouts = layouts[: args.limit]

    out_path = Path(args.out) if args.out else Path(f"evaluation_{args.method}.png")

    if args.method == "id3":
        agent = ID3Agent.from_layouts(all_layouts)
    elif args.method == "monkey":
        # Silence search-node progress output during evaluation; keep it only in precompute.py.
        cfg = replace(MonkeyConfig(), progress_enabled=False)
        agent = MonkeyAgent.from_layouts(all_layouts, top_k=int(cfg.top_k), cfg=cfg)
    else:
        if args.method == "elim":
            agent = ElimAgent.from_layouts(all_layouts)
        else:
            cfg = MCTSConfig()
            if args.mcts_sims is not None:
                cfg = MCTSConfig(
                    num_simulations=int(args.mcts_sims),
                    max_depth=cfg.max_depth,
                    c_ucb=cfg.c_ucb,
                    progressive_widening_k=cfg.progressive_widening_k,
                    rollout_depth=cfg.rollout_depth,
                    seed=cfg.seed,
                )
            if args.mcts_depth is not None:
                cfg = MCTSConfig(
                    num_simulations=cfg.num_simulations,
                    max_depth=int(args.mcts_depth),
                    c_ucb=cfg.c_ucb,
                    progressive_widening_k=cfg.progressive_widening_k,
                    rollout_depth=cfg.rollout_depth,
                    seed=cfg.seed,
                )
            agent = MCTSAgent.from_layouts(all_layouts, cfg=cfg)

    env = BombPlanesEnv(layouts=all_layouts, reward_mode="sparse", illegal_action="raise", max_steps=500)

    steps_list: list[int] = []
    start_t = time.time()
    last_print_t = start_t
    total = len(layouts)
    for i, layout in enumerate(layouts):
        steps = agent.play_one(env, layout=layout)
        steps_list.append(int(steps))

        # progress bar (throttled)
        now = time.time()
        if (i == 0) or (i + 1 == total) or (now - last_print_t >= 0.5):
            done = i + 1
            frac = done / total if total else 1.0
            bar_w = 30
            filled = int(bar_w * frac)
            bar = "=" * filled + " " * (bar_w - filled)
            elapsed = now - start_t
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("inf")
            eta_str = f"{eta:6.1f}s" if np.isfinite(eta) else "  inf s"
            # running stats
            sofar = np.asarray(steps_list, dtype=np.float64)
            mean_sofar = float(np.mean(sofar)) if sofar.size else 0.0
            median_sofar = float(np.median(sofar)) if sofar.size else 0.0
            msg = (
                f"\r[{bar}] {frac*100:6.2f}%  {done}/{total}  "
                f"mean {mean_sofar:6.2f}  median {median_sofar:6.2f}  "
                f"elapsed {elapsed:6.1f}s  eta {eta_str}"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()
            last_print_t = now

    sys.stdout.write("\n")

    steps_arr = np.array(steps_list, dtype=np.int32)
    mean_steps = float(np.mean(steps_arr))
    median_steps = float(np.median(steps_arr))

    print("=== Evaluation ===")
    print(f"method: {args.method}")
    print(f"layouts: {len(layouts)}")
    print(f"mean_steps: {mean_steps:.3f}")
    print(f"median_steps: {median_steps:.3f}")

    if plt is None:
        print("matplotlib not installed; skipping histogram plot. Install it to enable chart output.")
        return

    # "柱状图" as histogram of steps
    plt.figure(figsize=(12, 6))
    bins = np.arange(int(steps_arr.min()), int(steps_arr.max()) + 2) - 0.5
    plt.hist(steps_arr, bins=bins, edgecolor="black", alpha=0.75)
    plt.axvline(mean_steps, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_steps:.2f}")
    plt.axvline(median_steps, color="green", linestyle="--", linewidth=2, label=f"Median: {median_steps:.2f}")
    plt.title(f"Steps to hit all 3 heads ({args.method})")
    plt.xlabel("Steps")
    plt.ylabel("Count (layouts)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {out_path}")


if __name__ == "__main__":
    main()


