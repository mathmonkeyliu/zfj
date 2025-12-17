from __future__ import annotations

import sys
import argparse
import random
from dataclasses import replace

import numpy as np

# 允许以脚本方式运行：`python monkey/verify_id3_equivalence.py`
_THIS_DIR = __import__("pathlib").Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if sys.path and __import__("pathlib").Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(_ROOT_DIR))

from config import GRID_SIZE  # noqa: E402
from environment import build_outcome_table, load_layouts
from id3 import ID3Agent
from monkey import MonkeyConfig
from monkey.search import SearchContext, make_progress, _pick_topk_actions  # type: ignore
from monkey.symmetry import SymmetryGroup


def _filter_candidates(outcomes: np.ndarray, cand_idx: np.ndarray, shots: dict[int, int]) -> np.ndarray:
    idx = cand_idx
    for a, v in shots.items():
        col = outcomes[idx, int(a)]
        idx = idx[col == int(v)]
        if idx.size == 0:
            break
    return idx


def _random_reachable_state(
    *,
    outcomes: np.ndarray,
    layout_i: int,
    steps: int,
    rng: random.Random,
) -> dict[int, int]:
    """
    从某个真实布局出发，随机点若干格子并记录真实反馈，得到一个“必然可达”的状态 shots。
    """
    shots: dict[int, int] = {}
    available = list(range(GRID_SIZE * GRID_SIZE))
    rng.shuffle(available)
    for a in available[:steps]:
        v = int(outcomes[int(layout_i), int(a)])
        shots[int(a)] = int(v)
    return shots


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Monkey(topk=1) action selection matches ID3 exactly.")
    ap.add_argument("--cases", type=int, default=200, help="Number of random states to test.")
    ap.add_argument("--max-shots", type=int, default=12, help="Max number of observed shots in a random state.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layouts = load_layouts(None)
    outcomes, label_ids, labels = build_outcome_table(layouts)

    id3 = ID3Agent(outcomes=outcomes, label_ids=label_ids, labels=labels)

    cfg = replace(MonkeyConfig(), top_k=1, top_k_when_small_candidates=1, progress_enabled=False)
    sym = SymmetryGroup.build()
    ctx = SearchContext(
        outcomes=outcomes,
        label_ids=label_ids,
        labels=labels,
        cfg=cfg,
        sym=sym,
        progress=make_progress(cfg, enabled=False),
        value_cache={},
        best_action_cache={},
    )

    rng = random.Random(int(args.seed))
    base_idx = np.arange(outcomes.shape[0], dtype=np.int32)

    for t in range(int(args.cases)):
        layout_i = rng.randrange(0, outcomes.shape[0])
        k = rng.randrange(0, int(args.max_shots) + 1)
        shots = _random_reachable_state(outcomes=outcomes, layout_i=layout_i, steps=k, rng=rng)

        cand_idx = _filter_candidates(outcomes, base_idx, shots)
        if cand_idx.size == 0:
            # 理论不应出现（reachable state），但保险起见跳过
            continue

        unshot = np.ones((GRID_SIZE * GRID_SIZE,), dtype=bool)
        for a in shots.keys():
            unshot[int(a)] = False

        # ID3 best action
        a_id3 = id3._best_action(cand_idx=cand_idx, unshot=unshot)  # noqa: SLF001

        # Monkey action selection when topk=1 (should match ID3 exactly)
        acts = _pick_topk_actions(ctx, cand_idx, unshot, shots)
        a_monkey = int(acts[0]) if acts else -1

        if int(a_id3) != int(a_monkey):
            print("Mismatch!")
            print(f"case={t} layout_i={layout_i} shots={sorted(shots.items())}")
            print(f"id3={a_id3} monkey={a_monkey}")
            raise SystemExit(1)

    print("OK: all tested states matched (Monkey topk=1 == ID3).")


if __name__ == "__main__":
    main()


