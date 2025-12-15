"""
Root-level interactive CLI for Bombing Planes.

You choose an algorithm (ID3 / C4.5). The algorithm proposes the next coordinate to attack.
You then input the outcome:
  0 = MISS, 1 = BODY, 2 = HEAD
The session ends when all 3 heads are hit.

This program does NOT know the true layout. It maintains a candidate set over all layouts
from `layouts.jsonl` and filters by your feedback, exactly like a real interactive game.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Literal

import numpy as np

from config import GRID_SIZE, GridState
from environment import build_outcome_table, load_layouts


# --- ANSI colors ---
C_RESET = "\033[0m"
C_GRAY = "\033[90m"  # MISS
C_GREEN = "\033[92m"  # BODY
C_RED = "\033[91m"  # HEAD
C_YELLOW = "\033[93m"  # suggestion
C_BLUE = "\033[94m"  # frame


Algo = Literal["id3", "c45", "elim"]


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def draw_board(board_state: list[list[GridState]], suggestion: tuple[int, int] | None = None) -> None:
    clear_screen()
    print(f"{C_BLUE}--- 炸飞机 交互式 AI ---{C_RESET}")
    # columns header
    print(f"{C_BLUE} " + " ".join([str(i) for i in range(GRID_SIZE)]) + f"{C_RESET}")
    print(f"{C_BLUE}  +" + "-" * (GRID_SIZE * 2 - 1) + f"+{C_RESET}")

    for x in range(GRID_SIZE):  # x = row
        row_str = f"{C_BLUE}{x} |{C_RESET}"
        for y in range(GRID_SIZE):  # y = col
            cell = board_state[x][y]
            symbol = "·"
            color = C_RESET

            if suggestion is not None and (x, y) == suggestion:
                symbol = "?"
                color = C_YELLOW
            elif cell == GridState.MISS:
                symbol = "■"
                color = C_GRAY
            elif cell == GridState.BODY:
                symbol = "X"
                color = C_GREEN
            elif cell == GridState.HEAD:
                symbol = "H"
                color = C_RED

            row_str += f"{color} {symbol}{C_RESET}"
        print(row_str)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def _best_action_id3(outcomes: np.ndarray, label_ids: np.ndarray, cand_idx: np.ndarray, unshot: np.ndarray) -> int:
    y = label_ids[cand_idx]
    uniq, inv = np.unique(y, return_inverse=True)
    m = int(uniq.size)
    base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m))

    features = np.flatnonzero(unshot)
    best_gain = -1.0
    best_a = int(features[0])
    best_head_prob = -1.0

    inv64 = inv.astype(np.int64, copy=False)
    for a in features:
        col = outcomes[cand_idx, int(a)].astype(np.int64, copy=False)
        combo = col * m + inv64
        cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
        outcome_counts = cont.sum(axis=1)
        n = int(outcome_counts.sum())
        if n == 0:
            continue

        cond_entropy = 0.0
        for v in range(3):
            nv = int(outcome_counts[v])
            if nv == 0:
                continue
            cond_entropy += (nv / n) * _entropy_from_counts(cont[v])

        gain = base_entropy - cond_entropy
        head_prob = float(outcome_counts[2] / n)
        if (gain > best_gain) or (np.isclose(gain, best_gain) and head_prob > best_head_prob):
            best_gain = float(gain)
            best_a = int(a)
            best_head_prob = head_prob

    return best_a


def _best_action_c45(outcomes: np.ndarray, label_ids: np.ndarray, cand_idx: np.ndarray, unshot: np.ndarray) -> int:
    y = label_ids[cand_idx]
    uniq, inv = np.unique(y, return_inverse=True)
    m = int(uniq.size)
    base_entropy = _entropy_from_counts(np.bincount(inv, minlength=m))

    features = np.flatnonzero(unshot)
    best_ratio = -1.0
    best_a = int(features[0])
    best_head_prob = -1.0

    inv64 = inv.astype(np.int64, copy=False)
    for a in features:
        col = outcomes[cand_idx, int(a)].astype(np.int64, copy=False)
        combo = col * m + inv64
        cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
        outcome_counts = cont.sum(axis=1)
        n = int(outcome_counts.sum())
        if n == 0:
            continue

        cond_entropy = 0.0
        for v in range(3):
            nv = int(outcome_counts[v])
            if nv == 0:
                continue
            cond_entropy += (nv / n) * _entropy_from_counts(cont[v])

        info_gain = base_entropy - cond_entropy
        split_info = _entropy_from_counts(outcome_counts.astype(np.float64))
        if split_info <= 0:
            continue

        ratio = float(info_gain / split_info)
        head_prob = float(outcome_counts[2] / n)
        if (ratio > best_ratio) or (np.isclose(ratio, best_ratio) and head_prob > best_head_prob):
            best_ratio = ratio
            best_a = int(a)
            best_head_prob = head_prob

    return best_a


def _best_action_elim(outcomes: np.ndarray, label_ids: np.ndarray, cand_idx: np.ndarray, unshot: np.ndarray) -> int:
    """
    排除法（minimax）：
    对每个未点格子 a，分别假设反馈 v∈{0,1,2}，过滤候选后看剩余“机头分布(label)”的种类数。
    取三种反馈中的最坏情况（最大值），选择让该最坏情况最小的 a。

    次级排序：expected_unique_labels 更小；再其次：HEAD 概率更高；最后：动作 id 更小。
    """
    y = label_ids[cand_idx]
    uniq, inv = np.unique(y, return_inverse=True)
    m = int(uniq.size)

    features = np.flatnonzero(unshot)
    best_a = int(features[0])
    best_worst = 1 << 30
    best_expected = float("inf")
    best_head_prob = -1.0

    inv64 = inv.astype(np.int64, copy=False)
    for a in features:
        col = outcomes[cand_idx, int(a)].astype(np.int64, copy=False)
        combo = col * m + inv64
        cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
        outcome_counts = cont.sum(axis=1).astype(np.float64, copy=False)
        n = float(outcome_counts.sum())
        if n <= 0:
            continue

        uniq_per_outcome = (cont > 0).sum(axis=1).astype(np.float64, copy=False)
        worst = int(uniq_per_outcome.max())
        probs = outcome_counts / n
        expected = float((probs * uniq_per_outcome).sum())
        head_prob = float(probs[2])

        if (
            (worst < best_worst)
            or (worst == best_worst and expected < best_expected - 1e-12)
            or (worst == best_worst and np.isclose(expected, best_expected) and head_prob > best_head_prob + 1e-12)
            or (
                worst == best_worst
                and np.isclose(expected, best_expected)
                and np.isclose(head_prob, best_head_prob)
                and int(a) < best_a
            )
        ):
            best_worst = worst
            best_expected = expected
            best_head_prob = head_prob
            best_a = int(a)

    return best_a


def interactive_game(algo: Algo, layouts_file: str | None = None) -> None:
    layouts = load_layouts(layouts_file)
    outcomes, label_ids, labels = build_outcome_table(layouts)

    # board_state[x][y] where x=row, y=col
    board_state = [[GridState.UNKNOWN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    unshot = np.ones((GRID_SIZE * GRID_SIZE,), dtype=bool)
    cand_idx = np.arange(outcomes.shape[0], dtype=np.int32)

    heads_hit = 0

    print(f"{C_GREEN}模型载入完毕。初始可能性空间: {len(cand_idx)} 种布局{C_RESET}")
    try:
        input(f"{C_BLUE}按回车开始 > {C_RESET}")
    except EOFError:
        print("\nEOF received, exiting.")
        return

    while True:
        draw_board(board_state)

        if heads_hit >= 3:
            print(f"\n{C_RED}恭喜！已击落全部 3 个机头！{C_RESET}")
            break

        if cand_idx.size == 0:
            print(f"\n{C_RED}没有符合条件的布局！请检查输入是否有误。{C_RESET}")
            break

        # terminal rule: if only one head-set label remains, directly suggest remaining heads
        possible_labels = np.unique(label_ids[cand_idx])
        suggestion: tuple[int, int] | None = None

        if possible_labels.size == 1:
            heads = labels[int(possible_labels[0])]
            for hx, hy in heads:
                a = int(hx) * GRID_SIZE + int(hy)
                if unshot[a]:
                    suggestion = (int(hx), int(hy))
                    break

        start_time = time.time()
        if suggestion is None:
            if not unshot.any():
                print(f"\n{C_RED}没有可用动作了（所有格子都被点过）。{C_RESET}")
                break
            if algo == "id3":
                a = _best_action_id3(outcomes, label_ids, cand_idx, unshot)
            elif algo == "c45":
                a = _best_action_c45(outcomes, label_ids, cand_idx, unshot)
            else:
                a = _best_action_elim(outcomes, label_ids, cand_idx, unshot)
            suggestion = divmod(int(a), GRID_SIZE)  # (row, col)
        calc_time = time.time() - start_time

        draw_board(board_state, suggestion=suggestion)
        print(f"\n算法: {C_YELLOW}{algo}{C_RESET} | 已击落机头: {heads_hit}/3 | 剩余可能性: {C_RED}{cand_idx.size}{C_RESET}")
        print(f"AI 建议打击坐标: {C_YELLOW}{list(suggestion)}{C_RESET} (计算耗时: {calc_time:.2f}s)")
        print("请输入结果: 0=未击中, 1=机身, 2=机头")

        try:
            res_str = input(f"{C_BLUE}输入结果 (0/1/2) > {C_RESET}").strip()
        except EOFError:
            print("\nEOF received, exiting.")
            return
        if res_str not in {"0", "1", "2"}:
            print(f"{C_RED}非法输入: {res_str}，请重新输入。{C_RESET}")
            try:
                input(f"{C_BLUE}按回车继续 > {C_RESET}")
            except EOFError:
                print("\nEOF received, exiting.")
                return
            continue

        x, y = suggestion  # x=row, y=col
        a = x * GRID_SIZE + y
        if not unshot[a]:
            print(f"{C_RED}这个格子已经点过了：{suggestion}，请检查。{C_RESET}")
            try:
                input(f"{C_BLUE}按回车继续 > {C_RESET}")
            except EOFError:
                print("\nEOF received, exiting.")
                return
            continue

        # update board and counters
        if res_str == "0":
            obs_v = 0
            board_state[x][y] = GridState.MISS
        elif res_str == "1":
            obs_v = 1
            board_state[x][y] = GridState.BODY
        else:
            obs_v = 2
            board_state[x][y] = GridState.HEAD
            heads_hit += 1

        unshot[a] = False

        # filter candidate layouts by consistency with this observation
        col = outcomes[cand_idx, int(a)]
        cand_idx = cand_idx[col == obs_v]

        print(f"收到反馈 {list(suggestion)} -> {res_str}，剩余可能性: {C_RED}{cand_idx.size}{C_RESET}")
        try:
            input(f"{C_BLUE}按回车继续 > {C_RESET}")
        except EOFError:
            print("\nEOF received, exiting.")
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Bombing Planes: you provide outcomes, AI suggests next move.")
    ap.add_argument("--algo", choices=["id3", "c45", "elim"], default=None, help="Algorithm to use (or choose interactively).")
    ap.add_argument("--layouts-file", default=None, help="Path to layouts.jsonl (default from config.LAYOUT_FILE).")
    args = ap.parse_args()

    algo: Algo
    if args.algo is None:
        print("请选择算法：")
        print("  1) ID3 (信息增益)")
        print("  2) C4.5 (增益率)")
        print("  3) 排除法 (minimax 最坏情况下剩余机头分布数最小)")
        try:
            c = input("输入 1/2/3 > ").strip()
        except EOFError:
            print("\nEOF received, exiting.")
            return
        if c == "1":
            algo = "id3"
        elif c == "2":
            algo = "c45"
        else:
            algo = "elim"
    else:
        algo = args.algo  # type: ignore[assignment]

    interactive_game(algo=algo, layouts_file=args.layouts_file)


if __name__ == "__main__":
    main()


