#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from aircraft_ai.agent import Agent, ReplayBuffer, Transition
from aircraft_ai.env import AircraftEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练炸飞机DQN智能体")
    parser.add_argument("--episodes", type=int, default=4000, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="每次优化的批大小")
    parser.add_argument("--replay-size", type=int, default=50_000, help="经验回放容量")
    parser.add_argument("--hidden-dim", type=int, default=256, help="网络隐藏层维度")
    parser.add_argument("--gamma", type=float, default=0.98, help="折扣因子")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="起始探索率")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="最低探索率")
    parser.add_argument("--epsilon-decay", type=float, default=0.999, help="每步的探索率衰减因子")
    parser.add_argument("--target-update", type=int, default=200, help="目标网络同步间隔")
    parser.add_argument("--eval-interval", type=int, default=400, help="评估间隔")
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/aircraft_dqn.pt"), help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    return parser.parse_args()


def evaluate(env: AircraftEnv, agent: Agent, runs: int = 64) -> Tuple[float, float]:
    rewards = []
    steps = []
    for _ in range(runs):
        obs = env.reset()
        done = False
        total_reward = 0.0
        moves = 0
        while not done:
            mask = env.action_mask()
            action = agent.select_action(obs, mask, epsilon=0.0)
            feedback = env.step(action)
            obs = feedback.observation
            total_reward += feedback.reward
            moves += 1
            done = feedback.done
        rewards.append(total_reward)
        steps.append(moves)
    return statistics.mean(rewards), statistics.mean(steps)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    env = AircraftEnv(seed=args.seed)
    eval_env = AircraftEnv(seed=args.seed + 1)
    agent = Agent(device=device, hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma)
    replay = ReplayBuffer(capacity=args.replay_size)

    epsilon = args.epsilon_start
    global_step = 0

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        moves = 0

        while not done:
            mask = env.action_mask()
            action = agent.select_action(obs, mask, epsilon)
            feedback = env.step(action)
            next_obs = feedback.observation
            next_mask = env.action_mask()

            transition = Transition(
                state=obs.copy(),
                action=action,
                reward=feedback.reward,
                next_state=next_obs.copy(),
                done=feedback.done,
                next_mask=next_mask.copy(),
            )
            replay.push(transition)

            obs = next_obs
            done = feedback.done
            episode_reward += feedback.reward
            moves += 1
            global_step += 1

            if len(replay) >= args.batch_size:
                agent.optimize(replay, args.batch_size)

            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % args.target_update == 0:
            agent.hard_update()

        eval_reward = float("nan")
        eval_steps = float("nan")
        if episode % args.eval_interval == 0:
            eval_reward, eval_steps = evaluate(eval_env, agent)

        print(
            f"Episode {episode:05d} | steps={moves:3d} | reward={episode_reward:6.2f} | "
            f"epsilon={epsilon:.3f} | eval_reward={eval_reward:.2f} | eval_steps={eval_steps:.1f}"
        )

    agent.save(args.save_path)
    print(f"模型已保存至 {args.save_path}")


if __name__ == "__main__":
    main()
