import argparse
import time
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import GRID_SIZE, GridState
from id3 import ID3Agent
from utils import decode_layouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=None, help="number of layouts to evaluate")
    parser.add_argument("--out", type=str, required=True, help="Output path for the chart (.png)")
    parser.add_argument("--method", type=str, default="id3", choices=["id3", "min_avg"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to min_avg checkpoint (.json)")
    args = parser.parse_args()

    all_layouts = [layout for grids in decode_layouts().values() for layout in grids]
    layouts_to_test = all_layouts[: args.num] if args.num else all_layouts

    steps_list = []
    start_time = time.time()
    
    if args.method == "id3":
        agent = ID3Agent()
        pbar = tqdm(layouts_to_test, desc="Evaluating id3", total=len(layouts_to_test), unit="layout")
        for layout in pbar:
            observed = np.full(GRID_SIZE * GRID_SIZE, GridState.UNKNOWN, dtype=np.uint8)
            steps = 0
            while True:
                moves = agent.select_move(observed)
                if not moves:
                    break
                move = moves[0]
                steps += 1
                observed[move] = layout[move]
            steps_list.append(steps)
            if len(steps_list) % 10 == 0 or len(steps_list) == len(layouts_to_test):
                step_mean = sum(steps_list) / len(steps_list) if steps_list else 0
                pbar.set_postfix(mean=f"{step_mean:.3f}")
    
    elif args.method == "min_avg":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for min_avg method")
        with open(args.checkpoint, "r") as f:
            raw_policy = json.load(f)
        
        policy = {}
        for key, move in raw_policy.items():
            state_tuple = tuple(map(int, list(key)))
            policy[state_tuple] = move

        pbar = tqdm(layouts_to_test, desc="Evaluating min_avg", total=len(layouts_to_test), unit="layout")
        for layout in pbar:
            observed = np.full(GRID_SIZE * GRID_SIZE, GridState.UNKNOWN, dtype=np.uint8)
            steps = 0
            while True:
                state_tuple = tuple(observed)
                if state_tuple not in policy:
                    steps += 3 - np.count_nonzero(observed == GridState.HEAD)
                    break
                move = policy[state_tuple]
                steps += 1
                observed[move] = layout[move]
            steps_list.append(steps)
            if len(steps_list) % 10 == 0 or len(steps_list) == len(layouts_to_test):
                step_mean = sum(steps_list) / len(steps_list) if steps_list else 0
                pbar.set_postfix(mean=f"{step_mean:.3f}")

    total_time = time.time() - start_time
    
    steps_arr = np.array(steps_list, dtype=np.int32)
    mean_steps = np.mean(steps_arr)
    median_steps = np.median(steps_arr)

    print("=== Evaluation Results ===")
    print(f"Method: {args.method}")
    print(f"Total Layouts: {len(layouts_to_test)}")
    print(f"Mean Steps: {mean_steps:.3f}")
    print(f"Median Steps: {median_steps:.3f}")
    print(f"Total Time: {total_time:.3f}s")
    print(f"Avg Time per Layout: {total_time/len(layouts_to_test):.3f}s")

    out_path = Path(args.out)
    
    plt.figure(figsize=(14, 7))
    
    min_steps = int(steps_arr.min())
    max_steps = int(steps_arr.max())
    bins = np.arange(min_steps, max_steps + 2) - 0.5
    
    counts, _, patches = plt.hist(steps_arr, bins=bins, edgecolor="black", alpha=0.75, color='steelblue')
    
    for count, patch in zip(counts, patches):
        if count > 0:
            height = patch.get_height()
            plt.text(patch.get_x() + patch.get_width() / 2., height, int(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.axvline(mean_steps, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_steps:.3f}")
    plt.axvline(median_steps, color="green", linestyle="--", linewidth=2, label=f"Median: {median_steps:.3f}")
    
    plt.xticks(range(min_steps, max_steps + 1))
    
    plt.title(f"Algorithm: {args.method}", fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("number of layouts", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart to: {out_path}")

if __name__ == "__main__":
    main()
