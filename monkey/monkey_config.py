from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonkeyConfig:
    """
    Monkey = minimax + alpha-beta pruning + ID3 information gain to rank actions (Top-K).

    你主要会改这些超参数：
    - top_k: 正常情况下每个 MIN 节点保留的候选动作数
    - small_candidates_threshold / top_k_when_small_candidates: 候选布局数量较小时适当加大 TopK
    """

    # --- action pruning ---
    top_k: int = 3
    small_candidates_threshold: int = 5
    # 默认保持与 ID3 完全一致：top_k=1 时不应“自动放大”
    top_k_when_small_candidates: int = 5

    # --- symmetry pruning (only for equal-gain actions) ---
    symmetry_enabled: bool = True
    # “信息增益相同”的判定：先 round 到固定小数位，再用该 key 做同一桶的对称去重
    symmetry_gain_round_ndigits: int = 6
    # 对称判定是否考虑当前状态（已观测到的格子必须在对称变换下保持不变）
    symmetry_consider_state: bool = True

    # --- search ---
    alpha_beta: bool = True
    # 预处理/调试时展示搜索进度条；evaluate/interactive 默认关闭（见调用处）
    progress_enabled: bool = True

    # --- progress estimate ---
    # 进度条总节点估计使用 (topk*3)^(depth/2)。depth 默认用“叶子平均深度”动态更新；
    # 如果 leaf_count=0 则回退到该 hint。
    progress_depth_hint: int = 14

    # --- stability / reproducibility ---
    # 当信息增益和 head 概率都相同，按 action id 升序保证稳定
    deterministic: bool = True


