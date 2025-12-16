"""
预计算搜索树并保存，带进度条。

使用方法：
    python -m monkey.precompute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from config import GRID_SIZE
from environment import load_layouts, build_outcome_table

from .agent import MonkeyAgent
from .config import MonkeyConfig


def estimate_total_nodes(top_k: int, avg_depth: int) -> int:
    """
    估计总节点数。
    
    每个节点有 top_k 个子节点（玩家选择），每个子节点有 3 个子节点（对手选择）。
    但由于 alpha-beta 剪枝，实际节点数大约是理论值的一半。
    
    总节点数 ≈ sum_{d=0}^{depth-1} (top_k * 3)^d / 2^d
              = sum_{d=0}^{depth-1} (top_k * 3 / 2)^d
    """
    if avg_depth <= 0:
        return 1
    
    ratio = top_k * 3.0 / 2.0
    if ratio <= 1:
        return int(avg_depth * 100)
    
    # 几何级数求和：sum_{d=0}^{n-1} r^d = (r^n - 1) / (r - 1)
    total = (ratio ** avg_depth - 1) / (ratio - 1)
    return int(total)


class ProgressTracker:
    """进度跟踪器，动态更新节点数估计"""
    
    def __init__(self, initial_estimate: int, top_k: int, enabled: bool = True):
        self.total_nodes = initial_estimate
        self.visited_nodes = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.enabled = enabled
        self.top_k = top_k
        
        # 深度统计
        self.depth_sum = 0
        self.depth_count = 0
        self.max_depth = 0
        self.current_depth = 0
    
    def update(self, delta: int = 1, depth: int = 0) -> None:
        """更新进度"""
        if not self.enabled:
            return
        
        self.visited_nodes += delta
        
        # 更新深度统计
        if depth > 0:
            self.depth_sum += depth
            self.depth_count += 1
            self.max_depth = max(self.max_depth, depth)
            self.current_depth = depth
            
            # 动态更新总节点数估计
            if self.depth_count > 100:  # 有足够的样本后才更新
                avg_depth = self.depth_sum / self.depth_count
                # 使用实际观察到的平均深度重新估计
                # 每层约有 top_k * 3 / 2 个节点（考虑 alpha-beta 剪枝）
                ratio = self.top_k * 3.0 / 2.0
                if ratio > 1:
                    estimated = int((ratio ** avg_depth - 1) / (ratio - 1))
                else:
                    estimated = int(avg_depth * 100)
                # 取当前估计和已访问节点数的最大值
                self.total_nodes = max(estimated, self.visited_nodes + 1000)
        
        now = time.time()
        
        # 每 0.5 秒更新一次
        if now - self.last_print_time >= 0.5 or self.visited_nodes >= self.total_nodes:
            self._print_progress()
            self.last_print_time = now
    
    def _print_progress(self) -> None:
        """打印进度条"""
        frac = min(1.0, self.visited_nodes / self.total_nodes) if self.total_nodes > 0 else 1.0
        bar_w = 40
        filled = int(bar_w * frac)
        bar = "=" * filled + ">" + " " * (bar_w - filled - 1) if filled < bar_w else "=" * bar_w
        
        elapsed = time.time() - self.start_time
        rate = self.visited_nodes / elapsed if elapsed > 0 else 0.0
        remaining = self.total_nodes - self.visited_nodes
        eta = remaining / rate if rate > 0 else float("inf")
        eta_str = f"{eta:6.1f}s" if np.isfinite(eta) and eta >= 0 else "  ???  "
        
        avg_depth_str = f"{self.depth_sum / self.depth_count:.1f}" if self.depth_count > 0 else "N/A"
        
        msg = (
            f"\r[{bar}] {frac*100:6.2f}%  "
            f"{self.visited_nodes:,}/{self.total_nodes:,} nodes  "
            f"depth {self.current_depth}/{self.max_depth} (avg {avg_depth_str})  "
            f"elapsed {elapsed:6.1f}s  eta {eta_str}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
    
    def finish(self) -> None:
        """完成进度"""
        if not self.enabled:
            return
        self.total_nodes = self.visited_nodes
        self._print_progress()
        sys.stdout.write("\n")
        sys.stdout.flush()


def precompute_tree(
    agent: MonkeyAgent,
    cfg: MonkeyConfig,
    output_path: Path,
) -> None:
    """
    预计算从初始状态开始的搜索树。
    
    搜索树格式：
    {
        state_key: (best_action, best_value, child_states),
        ...
    }
    
    其中 state_key = (sorted_cand_idx, unshot_mask, heads_hit)
    """
    print("开始预计算搜索树...")
    print(f"配置: top_k={cfg.top_k}, expanded_top_k={cfg.expanded_top_k}, expand_threshold={cfg.expand_threshold}")
    
    # 估计总节点数（假设平均搜索深度为 12）
    avg_depth = 12
    total_nodes = estimate_total_nodes(cfg.top_k, avg_depth)
    print(f"初始估计节点数: ~{total_nodes:,}（将根据实际搜索深度动态调整）")
    
    # 创建进度跟踪器
    progress = ProgressTracker(total_nodes, top_k=cfg.top_k, enabled=cfg.progress_enabled)
    
    # 搜索树缓存（使用字符串键以提高效率和 JSON 兼容性）
    search_tree: dict[str, tuple[int, int]] = {}
    
    # 从初始状态开始遍历
    initial_cand_idx = np.arange(len(agent.outcomes), dtype=np.int32)
    initial_unshot = np.ones(GRID_SIZE * GRID_SIZE, dtype=bool)
    initial_heads_hit = 0
    
    print("\n开始遍历搜索树...")
    
    # 包装 agent 的 minimax 方法以记录进度
    original_minimax = agent._minimax
    original_minimax_max = agent._minimax_max
    
    def tracked_minimax(*args, **kwargs):
        # 获取深度参数
        depth = args[3] if len(args) > 3 else kwargs.get('depth', 0)
        progress.update(1, depth=depth)
        result = original_minimax(*args, **kwargs)
        # 保存到搜索树
        if len(args) >= 3:
            cand_idx, unshot, heads_hit = args[0], args[1], args[2]
            state_key = agent._state_key(cand_idx, unshot, heads_hit)
            if result[0] is not None:
                search_tree[state_key] = (result[0], result[1])
        return result
    
    def tracked_minimax_max(*args, **kwargs):
        # 获取深度参数
        depth = args[3] if len(args) > 3 else kwargs.get('depth', 0)
        progress.update(1, depth=depth)
        return original_minimax_max(*args, **kwargs)
    
    # 临时替换方法
    agent._minimax = tracked_minimax
    agent._minimax_max = tracked_minimax_max
    
    try:
        # 执行一次完整搜索
        best_action, best_value = agent._minimax(
            initial_cand_idx,
            initial_unshot,
            initial_heads_hit,
            depth=0,
            alpha=cfg.initial_alpha,
            beta=cfg.initial_beta,
            is_min_player=True,
        )
        
        progress.finish()
        
        print(f"\n搜索完成！")
        print(f"初始最佳动作: {best_action}")
        print(f"最坏情况步数: {best_value}")
        print(f"搜索树大小: {len(search_tree):,} 个状态")
        print(f"总访问节点数: {progress.visited_nodes:,}")
        print(f"最大搜索深度: {progress.max_depth}")
        if progress.depth_count > 0:
            print(f"平均搜索深度: {progress.depth_sum / progress.depth_count:.2f}")
        
        # 保存搜索树（转换为 JSON 兼容格式）
        # 搜索树：dict[str, tuple[int, int]] -> dict[str, list[int, int]]
        search_tree_serializable = {k: list(v) for k, v in search_tree.items()}
        
        data = {
            "config": {
                "top_k": cfg.top_k,
                "expand_threshold": cfg.expand_threshold,
                "expanded_top_k": cfg.expanded_top_k,
            },
            "search_tree": search_tree_serializable,
            "initial_best_action": int(best_action) if best_action is not None else None,
            "initial_best_value": int(best_value),
            "tree_size": len(search_tree),
        }
        
        print(f"\n保存搜索树到: {output_path}")
        
        # 根据文件扩展名选择格式
        if output_path.suffix == '.json':
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            # 默认使用 JSON，但保持 .pkl 扩展名（实际上是 JSON 格式）
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        
        print(f"保存完成！文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    finally:
        # 恢复原始方法
        agent._minimax = original_minimax
        agent._minimax_max = original_minimax_max


def main() -> None:
    parser = argparse.ArgumentParser(description="预计算 Monkey agent 的搜索树")
    parser.add_argument("--output", type=str, default="monkey_tree.json", help="输出文件路径（建议使用 .json 扩展名）")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k 参数（默认从配置文件加载）")
    parser.add_argument("--expanded-top-k", type=int, default=None, help="扩展后的 Top-k 参数（默认从配置文件加载）")
    parser.add_argument("--expand-threshold", type=int, default=None, help="扩展阈值（默认从配置文件加载）")
    args = parser.parse_args()
    
    # 加载布局
    print("加载布局...")
    layouts = load_layouts(None)
    print(f"已加载 {len(layouts)} 个布局")
    
    # 创建配置，从默认配置开始
    cfg = MonkeyConfig(progress_enabled=True)
    
    # 如果用户提供了参数，则覆盖默认值
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.expanded_top_k is not None:
        cfg.expanded_top_k = args.expanded_top_k
    if args.expand_threshold is not None:
        cfg.expand_threshold = args.expand_threshold
    
    # 创建 agent
    print("创建 agent...")
    agent = MonkeyAgent.from_layouts(layouts, cfg=cfg)
    
    # 预计算搜索树
    output_path = Path(args.output)
    precompute_tree(agent, cfg, output_path)
    
    print("完成！")


if __name__ == "__main__":
    main()

