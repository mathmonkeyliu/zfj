import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))

from config import GRID_SIZE
from environment import load_layouts, build_outcome_table
from monkey2.monkey_config import MonkeyConfig
from monkey2.search import SearchContext, make_progress, solve_minimax_and_record_best_action
from monkey2.symmetry import SymmetryGroup


def main():
    # 1. Load Layouts
    print("Loading layouts...")
    layouts = load_layouts()
    print(f"Loaded {len(layouts)} layouts.")
    
    outcomes, label_ids, labels = build_outcome_table(layouts)

    # 2. Config
    # Load default config from monkey2/monkey_config.py
    # User can modify that file to change parameters.
    cfg = MonkeyConfig()
    
    # Optionally override via command line args if implemented, 
    # but for now we trust the file config.
    
    print(f"Config: {cfg}")

    # 3. Setup Context
    sym = SymmetryGroup.build()
    progress = make_progress(cfg, enabled=True)
    
    ctx = SearchContext(
        outcomes=outcomes,
        label_ids=label_ids,
        labels=labels,
        cfg=cfg,
        sym=sym,
        progress=progress,
        value_cache={},
        best_action_cache={},
    )

    # 4. Run Search
    print("Starting precomputation search...")
    start_t = time.time()
    
    # Initial State
    cand_idx = np.arange(len(layouts), dtype=np.int32)
    unshot = np.ones(GRID_SIZE * GRID_SIZE, dtype=bool)
    shots: dict[int, int] = {}

    try:
        avg_steps, _ = solve_minimax_and_record_best_action(ctx, cand_idx, unshot, shots, depth=0)
    finally:
        progress.done()

    elapsed = time.time() - start_t
    print(f"Search complete in {elapsed:.1f}s.")
    print(f"Estimated Average Steps (weighted): {avg_steps:.2f}")
    print(f"Policy size: {len(ctx.best_action_cache)} states.")

    # 5. Save Policy
    output_path = Path("monkey2_policy.json")
    
    policy_export = {}
    for h, action_id in ctx.best_action_cache.items():
        x, y = divmod(int(action_id), GRID_SIZE)
        policy_export[h] = [x, y]

    # Export config as dict
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__slots__}

    data = {
        "policy": policy_export,
        "config": cfg_dict
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        
    print(f"Policy saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()

