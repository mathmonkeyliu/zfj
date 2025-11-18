"""
训练动态规划AI
支持AI自我对弈学习
"""

import numpy as np
from game_env import PlaneGame
from dp_ai import DPAI
import time
import os


def train_ai(num_episodes: int = 1000, save_interval: int = 100, 
             model_path: str = "ai_model.pkl"):
    """
    训练AI
    
    Args:
        num_episodes: 训练轮数
        save_interval: 保存间隔
        model_path: 模型保存路径
    """
    print("=" * 50)
    print("开始训练动态规划AI")
    print("=" * 50)
    
    # 创建两个AI（可以是对手，也可以共享经验）
    ai1 = DPAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    ai2 = DPAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    
    # 尝试加载已有模型
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        ai1.load_model(model_path)
        ai2.load_model(model_path)
    
    wins1 = 0
    wins2 = 0
    total_steps = 0
    
    start_time = time.time()
    
    for episode in range(num_episodes):
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
            
            print(f"\n轮次 {episode + 1}/{num_episodes}")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  AI1胜率: {wins1 / (episode + 1) * 100:.1f}% ({wins1}胜)")
            print(f"  AI2胜率: {wins2 / (episode + 1) * 100:.1f}% ({wins2}胜)")
            print(f"  平均步数: {total_steps / (episode + 1):.1f}")
            print(f"  AI1状态数: {stats1['num_states']}, Q值数: {stats1['num_q_entries']}")
            print(f"  AI2状态数: {stats2['num_states']}, Q值数: {stats2['num_q_entries']}")
            
            # 保存模型（保存AI1的模型）
            ai1.save_model(model_path)
            
            # 逐渐降低探索率
            if ai1.epsilon > 0.01:
                ai1.epsilon *= 0.99
                ai2.epsilon *= 0.99
    
    # 最终保存
    ai1.save_model(model_path)
    print(f"\n训练完成！模型已保存到: {model_path}")
    
    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n最终统计:")
    print(f"  总轮次: {num_episodes}")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  AI1胜率: {wins1 / num_episodes * 100:.1f}%")
    print(f"  AI2胜率: {wins2 / num_episodes * 100:.1f}%")
    print(f"  平均步数: {total_steps / num_episodes:.1f}")


def train_with_self_play(num_episodes: int = 1000, save_interval: int = 100,
                        model_path: str = "ai_model.pkl"):
    """
    自我对弈训练（两个AI共享同一个模型）
    
    Args:
        num_episodes: 训练轮数
        save_interval: 保存间隔
        model_path: 模型保存路径
    """
    print("=" * 50)
    print("开始自我对弈训练")
    print("=" * 50)
    
    # 创建共享模型的AI
    ai = DPAI(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    
    # 尝试加载已有模型
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        ai.load_model(model_path)
    
    wins1 = 0
    wins2 = 0
    total_steps = 0
    
    start_time = time.time()
    
    for episode in range(num_episodes):
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
            
            print(f"\n轮次 {episode + 1}/{num_episodes}")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  玩家1胜率: {wins1 / (episode + 1) * 100:.1f}% ({wins1}胜)")
            print(f"  玩家2胜率: {wins2 / (episode + 1) * 100:.1f}% ({wins2}胜)")
            print(f"  平均步数: {total_steps / (episode + 1):.1f}")
            print(f"  状态数: {stats['num_states']}, Q值数: {stats['num_q_entries']}")
            
            # 保存模型
            ai.save_model(model_path)
            
            # 逐渐降低探索率
            if ai.epsilon > 0.01:
                ai.epsilon *= 0.99
    
    # 最终保存
    ai.save_model(model_path)
    print(f"\n训练完成！模型已保存到: {model_path}")
    
    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n最终统计:")
    print(f"  总轮次: {num_episodes}")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  玩家1胜率: {wins1 / num_episodes * 100:.1f}%")
    print(f"  玩家2胜率: {wins2 / num_episodes * 100:.1f}%")
    print(f"  平均步数: {total_steps / num_episodes:.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练炸飞机AI")
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--save-interval", type=int, default=100, help="保存间隔")
    parser.add_argument("--model", type=str, default="ai_model.pkl", help="模型路径")
    parser.add_argument("--self-play", action="store_true", help="使用自我对弈模式")
    
    args = parser.parse_args()
    
    if args.self_play:
        train_with_self_play(args.episodes, args.save_interval, args.model)
    else:
        train_ai(args.episodes, args.save_interval, args.model)

