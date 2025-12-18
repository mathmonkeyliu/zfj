from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

# 允许以脚本方式运行：`python monkey/shrink_policy.py`
# 这时 sys.path[0] 会是 monkey/，会导致 `import config` 错误命中 monkey/config.py。
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(_ROOT_DIR))

from config import GRID_SIZE  # noqa: E402
from environment import build_outcome_table, load_layouts  # noqa: E402
from monkey.monkey_config import MonkeyConfig  # noqa: E402
from monkey.search import SearchContext, make_progress, solve_minimax_and_record_best_action  # noqa: E402
from monkey.state_hash import state_hash_hex  # noqa: E402
from monkey.symmetry import SymmetryGroup  # noqa: E402


def _stable_dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    tmp.replace(path)


def _load_policy_actions(data: dict[str, Any]) -> dict[str, int]:
    """
    兼容两种 value：
    - [x,y]
    - action_id int
    """
    raw = data.get("policy")
    if not isinstance(raw, dict):
        raise ValueError("Invalid policy JSON: expected dict at key 'policy'.")
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list) and len(v) == 2:
            x, y = int(v[0]), int(v[1])
            out[k] = int(x) * GRID_SIZE + int(y)
        else:
            out[k] = int(v)
    return out


def _collect_reachable_policy(
    *,
    ctx: SearchContext,
    seed_policy: dict[str, int],
    fill_missing: bool,
) -> dict[str, int]:
    """
    从 root 出发，沿“最优动作”展开：
    - 每个 MIN 状态只走 policy[state] 这一条动作
    - 但对该动作的反馈分支（0/1/2）全部展开（只保留对应布局非空的分支）

    返回：所有可达 MIN 状态的子集 policy（state_hash -> action_id）
    """
    stack: list[tuple[np.ndarray, np.ndarray, dict[int, int]]] = [
        (np.arange(ctx.outcomes.shape[0], dtype=np.int32), np.ones((GRID_SIZE * GRID_SIZE,), dtype=bool), {}),
    ]
    seen: set[str] = set()
    out_policy: dict[str, int] = {}

    while stack:
        cand_i, unshot_i, shots_i = stack.pop()
        h = state_hash_hex(shots_i)
        if h in seen:
            continue
        seen.add(h)

        # terminal: 3 heads hit
        heads_hit = sum(1 for v in shots_i.values() if int(v) == 2)
        if heads_hit >= 3:
            continue

        # terminal: label fixed
        possible_labels = np.unique(ctx.label_ids[cand_i])
        if possible_labels.size == 1:
            continue

        if not bool(unshot_i.any()):
            continue

        a = seed_policy.get(h)
        if a is None and fill_missing:
            solve_minimax_and_record_best_action(ctx, cand_i, unshot_i, dict(shots_i), depth=0)
            a = ctx.best_action_cache.get(h)

        if a is None:
            continue

        a = int(a)
        if not bool(unshot_i[a]):
            continue

        out_policy[h] = a

        unshot2 = unshot_i.copy()
        unshot2[a] = False
        col = ctx.outcomes[cand_i, a]
        for v in (0, 1, 2):
            sub = cand_i[col == int(v)]
            if sub.size == 0:
                continue
            shots2 = dict(shots_i)
            shots2[a] = int(v)
            stack.append((sub, unshot2, shots2))

    return out_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Shrink Monkey policy JSON to only states reachable when always following its best action.")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="Input policy JSON path.")
    ap.add_argument("--out", dest="out_path", type=str, required=True, help="Output policy JSON path.")
    ap.add_argument(
        "--fill-missing",
        action="store_true",
        help="If a reachable state is missing from the input policy (e.g. alpha-beta pruned), compute its best action on demand and include it.",
    )
    ap.add_argument("--layouts", type=str, default=None, help="Optional layouts file path (defaults to config.LAYOUT_FILE).")
    args = ap.parse_args()

    in_p = Path(args.in_path)
    with in_p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid policy JSON: expected top-level dict.")

    seed_policy = _load_policy_actions(data)

    # cfg：优先从文件读，失败回退默认；并强制关闭进度条（脚本更安静）
    cfg = MonkeyConfig()
    cfg_raw = data.get("config")
    if isinstance(cfg_raw, dict):
        try:
            cfg = MonkeyConfig(**cfg_raw)  # type: ignore[arg-type]
        except TypeError:
            cfg = MonkeyConfig()
    cfg = replace(cfg, progress_enabled=False)

    layouts = load_layouts(args.layouts)
    outcomes, label_ids, labels = build_outcome_table(layouts)

    sym = SymmetryGroup.build()
    progress = make_progress(cfg, enabled=False)
    ctx = SearchContext(
        outcomes=outcomes,
        label_ids=label_ids,
        labels=labels,
        cfg=cfg,
        sym=sym,
        progress=progress,
        value_cache={},
        best_action_cache=dict(seed_policy),  # 直接用输入 policy 作为 seed
    )

    # root minimax_value：优先沿用输入；若 fill_missing 则必要时补算
    root_val: int | None = None
    root_raw = data.get("root")
    if isinstance(root_raw, dict) and "minimax_value" in root_raw:
        try:
            root_val = int(root_raw["minimax_value"])
        except Exception:
            root_val = None
    if root_val is None and bool(args.fill_missing):
        cand0 = np.arange(outcomes.shape[0], dtype=np.int32)
        unshot0 = np.ones((GRID_SIZE * GRID_SIZE,), dtype=bool)
        root_val = int(solve_minimax_and_record_best_action(ctx, cand0, unshot0, {}, depth=0))

    reduced_actions = _collect_reachable_policy(ctx=ctx, seed_policy=seed_policy, fill_missing=bool(args.fill_missing))

    reduced_xy: dict[str, list[int]] = {}
    for h, a in reduced_actions.items():
        x, y = divmod(int(a), GRID_SIZE)
        reduced_xy[str(h)] = [int(x), int(y)]

    ver = 1
    try:
        ver = int(data.get("version", 1))
    except Exception:
        ver = 1

    out: dict[str, Any] = {
        "version": int(ver),
        "config": asdict(cfg),
        "scope": "reachable",
        "note": "reachable policy: only MIN states reachable when always following the best action; branches over all feasible outcomes.",
        "root": {"state_hash": state_hash_hex({})},
        "policy": reduced_xy,
        "source": {
            "input": str(in_p),
            "fill_missing": bool(args.fill_missing),
            "original_states": int(len(seed_policy)),
        },
    }
    if root_val is not None:
        out["root"]["minimax_value"] = int(root_val)

    out_p = Path(args.out_path)
    _stable_dump_json(out_p, out)
    print(f"Saved reachable policy: {out_p}  (states: {len(reduced_xy)}; from: {len(seed_policy)}; fill_missing={bool(args.fill_missing)})")


if __name__ == "__main__":
    main()


