# train.py
import json
import torch
import torch.optim as optim
import numpy as np
import random
from game import BattleGame
from model import AlphaZeroNet
from consts import GRID_SIZE

# 超参数
LR = 1e-4
EPOCHS = 100
BATCH_SIZE = 32
GAMES_PER_EPOCH = 1000

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载布局库
    print("Loading layouts...")
    with open("layouts.jsonl", "r") as f:
        layouts = [json.loads(line) for line in f]
    print(f"Loaded {len(layouts)} layouts.")
    
    model = AlphaZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_steps = 0
        
        # 收集数据
        batch_states = []
        batch_targets = [] # 最佳动作应为：如果知道底牌，哪里有飞机打哪里
        batch_values = []
        
        # 模拟对局
        for _ in range(GAMES_PER_EPOCH):
            layout = random.choice(layouts)
            game = BattleGame(layout)
            state = game.get_state()
            done = False
            game_history = []
            
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                
                # 简单策略：训练初期使用 Epsilon-Greedy，后期完全依赖网络
                epsilon = max(0.1, 1.0 - epoch/20)
                
                if random.random() < epsilon:
                    valid = game.get_valid_moves()
                    valid_indices = np.where(valid == 1)[0]
                    action = np.random.choice(valid_indices)
                else:
                    with torch.no_grad():
                        log_p, v = model(state_tensor)
                    p = torch.exp(log_p).cpu().numpy()[0]
                    valid = game.get_valid_moves()
                    p = p * valid
                    if p.sum() == 0: p = valid
                    action = np.argmax(p)
                
                next_state, reward, done = game.step(action)
                game_history.append((state, action))
                state = next_state
            
            total_steps += game.steps
            
            # 构造训练数据 (简单版：蒙特卡洛策略梯度)
            # 胜利时，我们鼓励它之前做出的“击中”决策，抑制“未击中”决策
            # 为了Top级水平，这里使用更强的监督信号：
            # "在当前状态下，哪些格子是真的有飞机？" -> 这是Policy的目标
            
            true_board = np.array(layout['board'])
            target_policy = np.zeros(100)
            # 目标：所有含有飞机的格子（且未被打过的）都是正确答案
            # 这是一个监督学习任务：Input(View) -> Output(True Plane Locations)

            for s, a in game_history:

                # 当前视角
                batch_states.append(s)
                
                # 理想 Policy: 真实的飞机位置 (Label)
                # 注意：只关注还没被打的格子
                current_valid_mask = (s.flatten() == 0) # Unknowns
                heads_mask = (true_board.flatten() == 2)
                
                ideal_targets = heads_mask & current_valid_mask
                
                label = ideal_targets.astype(float)
                if label.sum() > 0:
                    label = label / label.sum()
                else:
                    # 如果都打完了(最后一步)，无所谓
                    label = np.zeros(100)
                
                batch_targets.append(label)
                
                # Value: 剩余飞机数越少越好，或者步数越少越好
                # 这里简化为：能不能赢 (总是能赢，所以Value拟合距离胜利的步数倒数)
                batch_values.append(1.0) # 占位，主要练Policy
                
            if len(batch_states) >= BATCH_SIZE:
                # 执行梯度下降
                states_t = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
                targets_t = torch.tensor(np.array(batch_targets), dtype=torch.float32).to(device)
                
                optimizer.zero_grad()
                p_pred, v_pred = model(states_t)
                
                # Loss: KL Divergence for Policy (Multilabel classification essentially)
                # 使得预测分布接近真实分布
                loss_p = -torch.sum(targets_t * p_pred, dim=1).mean()
                
                loss = loss_p
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_states, batch_targets, batch_values = [], [], []
        
        avg_steps = total_steps / GAMES_PER_EPOCH
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Avg Steps to Win={avg_steps:.2f}")
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"bombing_plane_v{epoch+1}.pth")

if __name__ == "__main__":
    train()