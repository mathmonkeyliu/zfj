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
from monkey import MonkeyAgent, MonkeyConfig

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a method across all layouts (metric: steps to hit all 3 heads).")
    ap.add_argument("--method", choices=["id3", "monkey"], default="id3")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of layouts to evaluate.")
    ap.add_argument("--out", default=None, help="Output png path.")
    ap.add_argument("--topk", type=int, default=None, help="Monkey: override top_k (branching factor).")
    ap.add_argument("--precomputed", type=str, default=None, help="Monkey: path to precomputed search tree file.")
    # monkey is configured via monkey/config.py (edit that file to tune).
    args = ap.parse_args()

    # Candidate universe for the online algorithm: all layouts from file.
    all_layouts = load_layouts(None)  # read from config.LAYOUT_FILE

    # Evaluation set: default = all layouts; or a prefix via --limit for quick tests.
    layouts = all_layouts
    if args.limit is not None:
        layouts = layouts[: args.limit]

    out_path = Path(args.out) if args.out else Path(f"evaluation_{args.method}.png")

    if args.method == "id3":
        agent = ID3Agent.from_layouts(all_layouts)
    else:
        # Prefer precomputed policy if present (evaluation must be fast).
        precomputed_path = args.precomputed
        if precomputed_path is None:
            for cand in ("monkey_policy.json", "monkey/policy.json"):
                if Path(cand).exists():
                    precomputed_path = cand
                    break

        # Silence search-node progress output during evaluation; keep it only in precompute.py.
        if precomputed_path:
            # 使用预计算的搜索树
            print(f"Loading precomputed search tree from {precomputed_path}")
            agent = MonkeyAgent.from_precomputed(precomputed_path, all_layouts)
        else:
            # 在线计算
            print(
                "WARNING: monkey without --precomputed will be very slow. "
                "Run: python monkey/precompute.py --out monkey_policy.json"
            )
            cfg = MonkeyConfig()
            if args.topk is not None:
                cfg = replace(cfg, top_k=int(args.topk))
            cfg = replace(cfg, progress_enabled=False)
            agent = MonkeyAgent.from_layouts(all_layouts, cfg=cfg)

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
    plt.figure(figsize=(14, 7))
    
    # 计算每个步数的频数
    min_steps = int(steps_arr.min())
    max_steps = int(steps_arr.max())
    bins = np.arange(min_steps, max_steps + 2) - 0.5
    
    # 绘制直方图
    counts, _, patches = plt.hist(steps_arr, bins=bins, edgecolor="black", alpha=0.75, color='steelblue')
    
    # 在每根柱子顶部标注数值
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:  # 只标注非零的柱子
            height = patch.get_height()
            plt.text(patch.get_x() + patch.get_width() / 2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 均值和中位数线
    plt.axvline(mean_steps, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_steps:.2f}")
    plt.axvline(median_steps, color="green", linestyle="--", linewidth=2, label=f"Median: {median_steps:.2f}")
    
    # 设置横坐标为整数
    plt.xticks(range(min_steps, max_steps + 1))
    
    plt.title(f"Steps to hit all 3 heads ({args.method})", fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Count (layouts)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {out_path}")


if __name__ == "__main__":
    main()


