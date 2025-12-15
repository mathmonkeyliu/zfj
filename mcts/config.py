from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCTSConfig:
    """
    MCTS/POMCP 超参数配置。

    说明：
    - 本项目是“隐藏布局 + 三值观测”的信息收集问题。我们用 POMCP 风格：
      每次 simulation 都会从当前候选集合里随机采样一个 layout（粒子），
      然后在树内/rollout 中用该 layout 产生确定观测。
    """

    # --- Search budget ---
    num_simulations: int = 300  # 每一步要跑多少次 simulation（越大越强但越慢）
    max_depth: int = 120  # 每次 simulation 最多向前看多少步（含 rollout）；120 >= 100，确保理论上能跑到终局

    # --- UCB/UCT ---
    c_ucb: float = 1.4  # UCB 探索常数

    # --- Expansion / Rollout ---
    progressive_widening_k: int = 32  # 每个 belief node 最多“逐步扩展”的动作数上限（根节点会默认放开到全部动作）
    rollout_depth: int = 120  # rollout 最长长度（建议 <= max_depth；若更大会自动截断）

    # --- Particle belief (随机取样) ---
    particle_count: int = 128  # 每次 simulation 采样多少个粒子来近似 belief
    rollout_action_k: int = 24  # rollout/启发式每步只在 top-k 候选动作上做评估（提速）

    # --- Randomness ---
    seed: int | None = None  # None 表示使用随机种子


