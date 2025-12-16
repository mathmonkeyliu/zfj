"""
Configuration for Monkey agent (minimax + alpha-beta pruning).
"""

from dataclasses import dataclass


@dataclass
class MonkeyConfig:
    """
    超参数配置
    
    参数说明：
    - top_k: 玩家(min)选择动作时考虑的熵减最大的前 k 个格子
    - expand_threshold: 当剩余候选布局数量小于等于此值时，增大 top_k
    - expanded_top_k: 剩余布局数量少时，top_k 扩大到的值
    - progress_enabled: 是否启用进度条（在 precompute 时启用，在 evaluate 和 interactive 时禁用）
    """
    # 玩家(min)选择动作时考虑的熵减最大的前 top_k 个格子
    top_k: int = 3
    
    # 当剩余候选布局数量小于等于 expand_threshold 时，增大 top_k
    expand_threshold: int = 10
    
    # 剩余布局数量少时，top_k 扩大到的值
    expanded_top_k: int = 5
    
    # 是否启用进度条（在 precompute 时启用，在 evaluate 和 interactive 时禁用）
    progress_enabled: bool = True
    
    # alpha-beta 剪枝的初始值
    initial_alpha: float = float('-inf')
    initial_beta: float = float('inf')

