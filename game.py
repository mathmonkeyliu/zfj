# game.py
import numpy as np
from consts import *

class BattleGame:
    def __init__(self, layout_data=None):
        self.reset(layout_data)

    def reset(self, layout_data):
        """
        layout_data: dict, contains 'board' (10x10 with 0,1,2)
        """
        self.true_board = np.array(layout_data['board'])
        # 玩家看到的面板：0=Unknown, 1=Miss, 2=Hit, 3=Kill
        # 为了神经网络方便，我们通常做成多个channel，这里简化为单层整数
        self.view_board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.steps = 0
        self.planes_left = NUM_PLANES
        self.done = False
        return self.get_state()

    def step(self, action):
        # action: 0-99
        x, y = action % GRID_SIZE, action // GRID_SIZE
        
        if self.view_board[y][x] != STATE_UNKNOWN:
            # 重复攻击无效位置，给予惩罚
            return self.get_state(), -1, self.done

        self.steps += 1
        reward = 0
        cell_value = self.true_board[y][x]

        if cell_value == 0: # Empty
            self.view_board[y][x] = STATE_MISS
            reward = -0.1 # 轻微的时间惩罚
        elif cell_value == 1: # Body
            self.view_board[y][x] = STATE_HIT
            reward = 1.0 # 击中奖励
        elif cell_value == 2: # Head
            self.view_board[y][x] = STATE_KILL
            self.planes_left -= 1
            reward = 5.0 # 击落奖励
            # 击落时，可以选做把飞机的其他部分自动标记出来（可选规则），
            # 这里按严格规则：只显示机头红，其他变绿（如果之前没打过）
        
        if self.planes_left == 0:
            self.done = True
            reward += 20 # 胜利大奖

        return self.get_state(), reward, self.done

    def get_state(self):
        # 返回当前观察到的棋盘
        return self.view_board.copy()
        
    def get_valid_moves(self):
        return (self.view_board.flatten() == STATE_UNKNOWN).astype(int)