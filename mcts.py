# mcts.py (Simplified as Neural Agent)
import torch
import numpy as np
from consts import GRID_SIZE

class AlphaZeroAgent:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def select_action(self, board, epsilon=0.0):
        # board: (10, 10) numpy array
        valid_moves = (board.flatten() == 0).astype(float) # 0 is Unknown
        
        if np.random.rand() < epsilon:
            # 随机探索
            probs = valid_moves / valid_moves.sum()
            action = np.random.choice(len(valid_moves), p=probs)
            return action
            
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_probs, value = self.model(board_tensor)
            
        probs = torch.exp(log_probs).cpu().numpy()[0]
        
        # Mask invalid moves (already shot)
        probs = probs * valid_moves
        
        # 归一化
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = valid_moves / valid_moves.sum()
            
        # 选择概率最高的点 (Greedy) 或者按概率采样 (Sampling)
        # 比赛模式建议 Greedy，训练模式建议 Sampling
        action = np.argmax(probs)
        return action