#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from aircraft_ai.alphazero_net import AlphaZeroNet
from aircraft_ai.constants import BOARD_SIZE, CELL_HEAD
from aircraft_ai.env import AircraftEnv
from aircraft_ai.mcts import MCTS


@dataclass
class SelfPlayData:
    """自我对弈数据"""
    state: np.ndarray
    mcts_policy: np.ndarray
    value: float  # 最终结果：1=获胜，-1=失败，0=平局


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Aircraft Battle AlphaZero Agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--self-play-games", type=int, default=100, help="Self-play games per iteration")
    parser.add_argument("--mcts-simulations", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.0, help="PUCT exploration constant")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for move selection")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--num-res-blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--num-channels", type=int, default=128, help="Number of channels in conv layers")
    parser.add_argument("--eval-games", type=int, default=20, help="Evaluation games")
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/alphazero.pt"), help="Model save path")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"), help="Checkpoint directory")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def self_play_game(
    network: AlphaZeroNet,
    env: AircraftEnv,
    mcts: MCTS,
    temperature: float = 1.0,
) -> List[SelfPlayData]:
    """执行一局自我对弈，返回训练数据"""
    data = []
    obs = env.reset()
    done = False
    game_history = []
    total_reward = 0.0
    
    while not done:
        mask = env.action_mask()
        if mask.sum() == 0:
            break
        
        # 使用MCTS搜索
        mcts.temperature = temperature
        action, mcts_policy = mcts.search(obs, mask)
        
        # 记录数据
        game_history.append((obs.copy(), mcts_policy.copy(), action))
        
        # 执行动作
        feedback = env.step(action)
        obs = feedback.observation
        done = feedback.done
        total_reward += feedback.reward
    
    # 确定游戏结果：如果完成游戏（击落所有飞机）则获胜
    # 根据最终reward判断：如果reward包含REWARD_COMPLETE则获胜
    result = 1.0 if done and total_reward > 10.0 else -1.0
    
    # 为每个状态分配结果
    for state, policy, _ in game_history:
        data.append(SelfPlayData(
            state=state,
            mcts_policy=policy,
            value=result,
        ))
    
    return data


def train_network(
    network: AlphaZeroNet,
    data: List[SelfPlayData],
    optimizer: optim.Optimizer,
    batch_size: int,
    epochs: int,
    l2_reg: float,
    device: torch.device,
) -> Tuple[float, float]:
    """训练网络"""
    if len(data) == 0:
        return 0.0, 0.0
    
    network.train()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    # 准备数据
    states = torch.tensor(np.stack([d.state for d in data]), dtype=torch.float32, device=device)
    policies = torch.tensor(np.stack([d.mcts_policy for d in data]), dtype=torch.float32, device=device)
    values = torch.tensor([d.value for d in data], dtype=torch.float32, device=device).unsqueeze(1)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    
    for epoch in range(epochs):
        # 随机打乱数据
        indices = torch.randperm(len(data), device=device)
        
        for i in range(0, len(data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states[batch_indices]
            batch_policies = policies[batch_indices]
            batch_values = values[batch_indices]
            
            # 前向传播
            pred_policies, pred_values = network(batch_states)
            
            # 计算损失
            # 策略损失：交叉熵（注意：这里使用KL散度更准确，但交叉熵更简单）
            policy_loss = -(batch_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
            
            # 价值损失：均方误差
            value_loss = criterion_value(pred_values, batch_values)
            
            # 总损失
            loss = policy_loss + value_loss
            
            # L2正则化
            l2_loss = sum(p.pow(2.0).sum() for p in network.parameters())
            loss += l2_reg * l2_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    avg_policy_loss = total_policy_loss / (epochs * (len(data) // batch_size + 1))
    avg_value_loss = total_value_loss / (epochs * (len(data) // batch_size + 1))
    
    return avg_policy_loss, avg_value_loss


def evaluate(network: AlphaZeroNet, env: AircraftEnv, mcts: MCTS, num_games: int = 20) -> float:
    """评估网络性能"""
    network.eval()
    wins = 0
    
    for _ in range(num_games):
        obs = env.reset()
        done = False
        moves = 0
        
        while not done and moves < 200:
            mask = env.action_mask()
            if mask.sum() == 0:
                break
            
            # 使用MCTS（温度=0，确定性选择）
            mcts.temperature = 0.0
            action, _ = mcts.search(obs, mask)
            
            feedback = env.step(action)
            obs = feedback.observation
            done = feedback.done
            moves += 1
        
        # 判断是否获胜（简化：如果完成游戏则获胜）
        if done:
            wins += 1
    
    return wins / num_games


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    
    # 创建网络
    network = AlphaZeroNet(
        board_size=10,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    ).to(device)
    
    # 创建checkpoint目录
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_latest = args.checkpoint_dir / "checkpoint_latest.pt"
    checkpoint_best = args.checkpoint_dir / "checkpoint_best.pt"
    
    # 恢复训练状态
    start_iteration = 1
    checkpoint = None
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        if "model_state_dict" in checkpoint:
            network.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 兼容旧格式（只有state_dict）
            network.load_state_dict(checkpoint)
        start_iteration = checkpoint.get("iteration", 1) + 1 if isinstance(checkpoint, dict) else 1
        print(f"Loaded checkpoint from {args.resume}, resuming from iteration {start_iteration}")
    elif checkpoint_latest.exists():
        checkpoint = torch.load(checkpoint_latest, map_location=device)
        if "model_state_dict" in checkpoint:
            network.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 兼容旧格式
            network.load_state_dict(checkpoint)
        start_iteration = checkpoint.get("iteration", 1) + 1 if isinstance(checkpoint, dict) else 1
        print(f"Loaded latest checkpoint from {checkpoint_latest}, resuming from iteration {start_iteration}")
    
    # 创建优化器
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    
    # 恢复优化器状态
    if checkpoint is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # 创建环境
    env = AircraftEnv(seed=args.seed)
    eval_env = AircraftEnv(seed=args.seed + 1)
    
    # 创建MCTS（需要step函数）
    # 注意：MCTS需要能够从任意状态执行动作
    # 这里我们创建一个临时环境来执行动作
    def step_fn(state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        """MCTS使用的step函数"""
        # 创建临时环境，从给定状态执行动作
        temp_env = AircraftEnv(seed=None)
        # 复制状态到临时环境
        # 注意：state是归一化的浮点数，需要转换回整数格式
        knowledge_2d = (state.reshape(BOARD_SIZE, BOARD_SIZE) * CELL_HEAD).astype(np.int8)
        temp_env.state.knowledge = knowledge_2d.copy()
        temp_env.state.board = env.state.board  # 共享同一个board（简化处理）
        temp_env.state.turns = env.state.turns
        temp_env.state.done = env.state.done
        
        # 执行动作
        feedback = temp_env.step(action)
        return feedback.observation, feedback.reward, feedback.done
    
    mcts = MCTS(
        network=network,
        step_fn=step_fn,
        c_puct=args.c_puct,
        num_simulations=args.mcts_simulations,
        temperature=args.temperature,
        device=device,
    )
    
    eval_mcts = MCTS(
        network=network,
        step_fn=step_fn,
        c_puct=args.c_puct,
        num_simulations=args.mcts_simulations,
        temperature=0.0,  # 评估时使用确定性选择
        device=device,
    )
    
    # 训练循环
    best_win_rate = 0.0
    if checkpoint_best.exists():
        checkpoint = torch.load(checkpoint_best, map_location=device)
        best_win_rate = checkpoint.get("win_rate", 0.0)
        print(f"Best win rate from checkpoint: {best_win_rate:.2%}")
    
    data_buffer = deque(maxlen=10000)  # 数据缓冲区
    
    for iteration in range(start_iteration, args.episodes + 1):
        # 1. 自我对弈生成数据
        print(f"\nIteration {iteration}/{args.episodes}")
        print("Generating self-play data...")
        
        iteration_data = []
        for game in tqdm(range(args.self_play_games), desc="Self-play"):
            game_data = self_play_game(network, env, mcts, temperature=args.temperature)
            iteration_data.extend(game_data)
            data_buffer.extend(game_data)
        
        print(f"Generated {len(iteration_data)} training samples")
        
        # 2. 训练网络
        print("Training network...")
        policy_loss, value_loss = train_network(
            network, list(data_buffer), optimizer, args.batch_size, args.epochs, args.l2_reg, device
        )
        print(f"Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
        
        # 3. 评估
        win_rate = 0.0
        if iteration % 10 == 0:
            print("Evaluating...")
            win_rate = evaluate(network, eval_env, eval_mcts, args.eval_games)
            print(f"Win rate: {win_rate:.2%}")
            
            # 保存最佳模型
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                network.eval()
                # 保存最佳checkpoint
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "win_rate": win_rate,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                }, checkpoint_best)
                # 同时保存到主路径（兼容旧代码）
                torch.save(network.state_dict(), args.save_path)
                print(f"Saved best model (win rate: {win_rate:.2%}) to {checkpoint_best}")
        
        # 4. 每个episode都保存最新checkpoint
        network.eval()
        torch.save({
            "iteration": iteration,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "win_rate": win_rate,
            "best_win_rate": best_win_rate,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }, checkpoint_latest)
        if iteration % 10 == 0:
            print(f"Saved latest checkpoint to {checkpoint_latest}")
    
    print(f"\nTraining completed! Best win rate: {best_win_rate:.2%}")
    print(f"Final model saved to {args.save_path}")
    print(f"Latest checkpoint: {checkpoint_latest}")
    print(f"Best checkpoint: {checkpoint_best}")


if __name__ == "__main__":
    main()

