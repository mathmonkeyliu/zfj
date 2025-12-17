from __future__ import annotations

import sys
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

# 允许以脚本方式运行：`python monkey/precompute.py`
# 这时 sys.path[0] 会是 monkey/，会导致 `import config` 错误命中 monkey/config.py。
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(_ROOT_DIR))

from config import GRID_SIZE  # noqa: E402
from environment import build_outcome_table, load_layouts
from monkey.monkey_config import MonkeyConfig
from monkey.search import SearchContext, make_progress, solve_minimax_and_record_best_action
from monkey.state_hash import state_hash_hex
from monkey.symmetry import SymmetryGroup


def _stable_dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute Monkey policy: state_hash -> best next move (x,y).")
    ap.add_argument("--out", type=str, default="monkey_policy.json", help="Output JSON path.")
    ap.add_argument("--topk", type=int, default=None, help="Override config.top_k")
    ap.add_argument("--topk-small", type=int, default=None, help="Override config.top_k_when_small_candidates")
    ap.add_argument("--small-threshold", type=int, default=None, help="Override config.small_candidates_threshold")
    ap.add_argument("--depth-hint", type=int, default=None, help="Progress estimate depth hint (for (topk*3)^(depth/2)).")
    args = ap.parse_args()

    layouts = load_layouts(None)
    outcomes, label_ids, labels = build_outcome_table(layouts)

    cfg = MonkeyConfig()
    if args.topk is not None:
        cfg = replace(cfg, top_k=int(args.topk))
    if args.topk_small is not None:
        cfg = replace(cfg, top_k_when_small_candidates=int(args.topk_small))
    if args.small_threshold is not None:
        cfg = replace(cfg, small_candidates_threshold=int(args.small_threshold))
    if args.depth_hint is not None:
        cfg = replace(cfg, progress_depth_hint=int(args.depth_hint))
    cfg = replace(cfg, progress_enabled=True)

    # 初始状态
    cand_idx = np.arange(outcomes.shape[0], dtype=np.int32)
    unshot = np.ones((GRID_SIZE * GRID_SIZE,), dtype=bool)
    shots: dict[int, int] = {}

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

    sys.setrecursionlimit(10000)
    try:
        v0 = solve_minimax_and_record_best_action(ctx, cand_idx, unshot, shots, depth=0)
    finally:
        progress.done()

    # 导出：hash -> (x,y)
    policy_xy: dict[str, list[int]] = {}
    for h, a in ctx.best_action_cache.items():
        x, y = divmod(int(a), GRID_SIZE)
        policy_xy[str(h)] = [int(x), int(y)]

    out = {
        "version": 1,
        "config": asdict(cfg),
        "note": "policy maps state_hash (shots list hashed) -> best next move (x,y). outcome: 0=MISS,1=BODY,2=HEAD",
        "root": {"state_hash": state_hash_hex({}), "minimax_value": int(v0)},
        "policy": policy_xy,
    }

    out_path = Path(args.out)
    _stable_dump_json(out_path, out)
    print(f"Saved precomputed policy: {out_path}  (states: {len(policy_xy)})")


if __name__ == "__main__":
    main()


