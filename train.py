"""
训练动态规划AI
支持AI自我对弈学习
"""

import numpy as np
from game_env import PlaneGame
from dp_ai import DPAI
import time
import os

# 尝试导入GPU相关库
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def train_ai(num_episodes: int = 1000, save_interval: int = 100, 
             model_path: str = "ai_model.pkl", use_gpu: bool = False):
    """
    训练AI
    
    Args:
        num_episodes: 训练轮数
        save_interval: 保存间隔
        model_path: 模型保存路径
        use_gpu: 是否使用GPU加速（注意：基于表格的方法GPU加速效果有限）
    """
    print("=" * 50)
    print("开始训练动态规划AI")
    print("=" * 50)
    
    # GPU支持检查
    if use_gpu:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
        elif CUPY_AVAILABLE:
            print("使用CuPy进行GPU加速（numpy操作）")
        else:
            print("警告: 未检测到GPU支持，将使用CPU训练")
            print("提示: 基于表格的Q-learning方法GPU加速效果有限，建议使用CPU")
            use_gpu = False
    else:
        print("使用CPU训练")
    
    # 创建两个AI（可以是对手，也可以共享经验）
    ai1 = DPAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    ai2 = DPAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    
    # 尝试加载已有模型
    start_episode = 0
    loaded_stats = {}
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        loaded_info1 = ai1.load_model(model_path)
        loaded_info2 = ai2.load_model(model_path)
        
        # 获取已训练的轮次
        start_episode = loaded_info1.get('episode', 0)
        loaded_stats = loaded_info1.get('training_stats', {})
        
        if start_episode > 0:
            print(f"从轮次 {start_episode} 继续训练...")
            # 恢复统计信息
            wins1 = loaded_stats.get('wins1', 0)
            wins2 = loaded_stats.get('wins2', 0)
            total_steps = loaded_stats.get('total_steps', 0)
        else:
            wins1 = 0
            wins2 = 0
            total_steps = 0
    else:
        wins1 = 0
        wins2 = 0
        total_steps = 0
    
    start_time = time.time()
    
    # 从保存的轮次继续训练
    for episode in range(start_episode, start_episode + num_episodes):
        game = PlaneGame()
        
        # 随机放置飞机
        game.place_planes_random(1)
        game.place_planes_random(2)
        
        step = 0
        max_steps = 200  # 防止无限循环
        
        while not game.is_terminal() and step < max_steps:
            # AI1行动
            if not game.is_terminal():
                result1, reward1 = ai1.train_step(game, 1)
                step += 1
            
            # AI2行动
            if not game.is_terminal():
                result2, reward2 = ai2.train_step(game, 2)
                step += 1
        
        # 统计
        winner = game.get_winner()
        if winner == 1:
            wins1 += 1
        elif winner == 2:
            wins2 += 1
        
        total_steps += step
        
        # 定期保存和输出
        if (episode + 1) % save_interval == 0:
            elapsed = time.time() - start_time
            stats1 = ai1.get_statistics()
            stats2 = ai2.get_statistics()
            
            current_episode = episode + 1
            total_episodes = start_episode + num_episodes
            print(f"\n轮次 {current_episode}/{total_episodes} (本次训练: {current_episode - start_episode}/{num_episodes})")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  AI1胜率: {wins1 / current_episode * 100:.1f}% ({wins1}胜)")
            print(f"  AI2胜率: {wins2 / current_episode * 100:.1f}% ({wins2}胜)")
            print(f"  平均步数: {total_steps / current_episode:.1f}")
            print(f"  AI1状态数: {stats1['num_states']}, Q值数: {stats1['num_q_entries']}")
            print(f"  AI2状态数: {stats2['num_states']}, Q值数: {stats2['num_q_entries']}")
            
            # 保存模型（保存AI1的模型，包含训练轮次和统计信息）
            training_stats = {
                'wins1': wins1,
                'wins2': wins2,
                'total_steps': total_steps
            }
            ai1.save_model(model_path, episode=current_episode, training_stats=training_stats)
            
            # 逐渐降低探索率
            if ai1.epsilon > 0.01:
                ai1.epsilon *= 0.99
                ai2.epsilon *= 0.99
    
    # 最终保存
    final_episode = start_episode + num_episodes
    final_training_stats = {
        'wins1': wins1,
        'wins2': wins2,
        'total_steps': total_steps
    }
    ai1.save_model(model_path, episode=final_episode, training_stats=final_training_stats)
    print(f"\n训练完成！模型已保存到: {model_path}")
    
    # 最终统计
    elapsed = time.time() - start_time
    total_episodes_trained = final_episode
    print(f"\n最终统计:")
    print(f"  总训练轮次: {total_episodes_trained} (本次训练: {num_episodes})")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  AI1胜率: {wins1 / total_episodes_trained * 100:.1f}%")
    print(f"  AI2胜率: {wins2 / total_episodes_trained * 100:.1f}%")
    print(f"  平均步数: {total_steps / total_episodes_trained:.1f}")


def train_with_self_play(num_episodes: int = 1000, save_interval: int = 100,
                        model_path: str = "ai_model.pkl", use_gpu: bool = False):
    """
    自我对弈训练（两个AI共享同一个模型）
    
    Args:
        num_episodes: 训练轮数
        save_interval: 保存间隔
        model_path: 模型保存路径
        use_gpu: 是否使用GPU加速（注意：基于表格的方法GPU加速效果有限）
    """
    print("=" * 50)
    print("开始自我对弈训练")
    print("=" * 50)
    
    # GPU支持检查
    if use_gpu:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
        elif CUPY_AVAILABLE:
            print("使用CuPy进行GPU加速（numpy操作）")
        else:
            print("警告: 未检测到GPU支持，将使用CPU训练")
            print("提示: 基于表格的Q-learning方法GPU加速效果有限，建议使用CPU")
            use_gpu = False
    else:
        print("使用CPU训练")
    
    # 创建共享模型的AI
    ai = DPAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    
    # 尝试加载已有模型
    start_episode = 0
    loaded_stats = {}
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        loaded_info = ai.load_model(model_path)
        
        # 获取已训练的轮次
        start_episode = loaded_info.get('episode', 0)
        loaded_stats = loaded_info.get('training_stats', {})
        
        if start_episode > 0:
            print(f"从轮次 {start_episode} 继续训练...")
            # 恢复统计信息
            wins1 = loaded_stats.get('wins1', 0)
            wins2 = loaded_stats.get('wins2', 0)
            total_steps = loaded_stats.get('total_steps', 0)
        else:
            wins1 = 0
            wins2 = 0
            total_steps = 0
    else:
        wins1 = 0
        wins2 = 0
        total_steps = 0
    
    start_time = time.time()
    
    # 从保存的轮次继续训练
    for episode in range(start_episode, start_episode + num_episodes):
        game = PlaneGame()
        
        # 随机放置飞机
        game.place_planes_random(1)
        game.place_planes_random(2)
        
        step = 0
        max_steps = 200
        
        while not game.is_terminal() and step < max_steps:
            # 玩家1行动（使用AI）
            if not game.is_terminal():
                result1, reward1 = ai.train_step(game, 1)
                step += 1
            
            # 玩家2行动（使用同一个AI）
            if not game.is_terminal():
                result2, reward2 = ai.train_step(game, 2)
                step += 1
        
        # 统计
        winner = game.get_winner()
        if winner == 1:
            wins1 += 1
        elif winner == 2:
            wins2 += 1
        
        total_steps += step
        
        # 定期保存和输出
        if (episode + 1) % save_interval == 0:
            elapsed = time.time() - start_time
            stats = ai.get_statistics()
            
            current_episode = episode + 1
            total_episodes = start_episode + num_episodes
            print(f"\n轮次 {current_episode}/{total_episodes} (本次训练: {current_episode - start_episode}/{num_episodes})")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  玩家1胜率: {wins1 / current_episode * 100:.1f}% ({wins1}胜)")
            print(f"  玩家2胜率: {wins2 / current_episode * 100:.1f}% ({wins2}胜)")
            print(f"  平均步数: {total_steps / current_episode:.1f}")
            print(f"  状态数: {stats['num_states']}, Q值数: {stats['num_q_entries']}")
            
            # 保存模型（包含训练轮次和统计信息）
            training_stats = {
                'wins1': wins1,
                'wins2': wins2,
                'total_steps': total_steps
            }
            ai.save_model(model_path, episode=current_episode, training_stats=training_stats)
            
            # 逐渐降低探索率
            if ai.epsilon > 0.01:
                ai.epsilon *= 0.99
    
    # 最终保存
    final_episode = start_episode + num_episodes
    final_training_stats = {
        'wins1': wins1,
        'wins2': wins2,
        'total_steps': total_steps
    }
    ai.save_model(model_path, episode=final_episode, training_stats=final_training_stats)
    print(f"\n训练完成！模型已保存到: {model_path}")
    
    # 最终统计
    elapsed = time.time() - start_time
    total_episodes_trained = final_episode
    print(f"\n最终统计:")
    print(f"  总训练轮次: {total_episodes_trained} (本次训练: {num_episodes})")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  玩家1胜率: {wins1 / total_episodes_trained * 100:.1f}%")
    print(f"  玩家2胜率: {wins2 / total_episodes_trained * 100:.1f}%")
    print(f"  平均步数: {total_steps / total_episodes_trained:.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练炸飞机AI")
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--save-interval", type=int, default=100, help="保存间隔")
    parser.add_argument("--model", type=str, default="ai_model.pkl", help="模型路径")
    parser.add_argument("--self-play", action="store_true", help="使用自我对弈模式")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速（注意：基于表格的方法GPU加速效果有限）")
    
    args = parser.parse_args()
    
    if args.self_play:
        train_with_self_play(args.episodes, args.save_interval, args.model, args.gpu)
    else:
        train_ai(args.episodes, args.save_interval, args.model, args.gpu)

