from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Allow running as a script: add repo root to sys.path so `import environment` works.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment import build_outcome_table, load_layouts
from monkey import MonkeyAgent, MonkeyConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute (warm) monkey minimax cache for a belief-state.")
    ap.add_argument("--layouts-file", default=None, help="Path to layouts.jsonl (default from config.LAYOUT_FILE).")
    # monkey is configured via monkey/config.py (edit that file to tune).
    args = ap.parse_args()

    layouts = load_layouts(args.layouts_file)
    outcomes, label_ids, labels = build_outcome_table(layouts)

    cfg = MonkeyConfig()
    agent = MonkeyAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg)

    cand_idx = np.arange(outcomes.shape[0], dtype=np.int32)
    unshot = np.ones((100,), dtype=bool)
    a = agent.choose_action(cand_idx=cand_idx, unshot_actions=unshot, heads_hit=0)
    print(f"best_action={a} (cache warmed at {cfg.cache_dir})")


if __name__ == "__main__":
    main()


