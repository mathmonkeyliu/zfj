"""
基于动态规划的AI算法
使用值迭代和策略迭代来学习最优策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from game_env import PlaneGame, AttackResult, CellState


class DPAI:
    """动态规划AI"""
    
    def __init__(self, board_size: int = 10, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        """
        初始化AI
        
        Args:
            board_size: 棋盘大小
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率（epsilon-greedy）
        """
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # 状态值函数 V(s) - 使用哈希表存储
        # 键：状态的字符串表示，值：状态价值
        self.value_function = {}
        
        # 状态-动作值函数 Q(s, a)
        # 键：(状态字符串, 动作)的元组，值：Q值
        self.q_function = {}
        
        # 访问计数（用于统计）
        self.visit_count = {}
        
        # 经验回放缓冲区
        self.experience_buffer = []
        self.buffer_size = 10000
    
    def state_to_string(self, attack_record: np.ndarray) -> str:
        """
        将状态转换为字符串（用于哈希）
        
        Args:
            attack_record: 攻击记录矩阵
            
        Returns:
            状态字符串
        """
        return attack_record.tobytes().hex()
    
    def get_state_key(self, attack_record: np.ndarray) -> str:
        """获取状态键"""
        return self.state_to_string(attack_record)
    
    def get_q_key(self, state_key: str, action: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
        """获取Q函数键"""
        return (state_key, action)
    
    def get_state_value(self, state_key: str) -> float:
        """获取状态价值"""
        if state_key not in self.value_function:
            # 初始化：根据已击中数量给予基础价值
            self.value_function[state_key] = 0.0
        return self.value_function[state_key]
    
    def get_q_value(self, state_key: str, action: Tuple[int, int]) -> float:
        """获取Q值"""
        q_key = self.get_q_key(state_key, action)
        if q_key not in self.q_function:
            # 初始化Q值
            self.q_function[q_key] = 0.0
        return self.q_function[q_key]
    
    def update_q_value(self, state_key: str, action: Tuple[int, int], 
                      reward: float, next_state_key: Optional[str] = None):
        """
        更新Q值（Q-learning）
        
        Args:
            state_key: 当前状态键
            action: 执行的动作
            reward: 获得的奖励
            next_state_key: 下一状态键（如果游戏未结束）
        """
        q_key = self.get_q_key(state_key, action)
        current_q = self.get_q_value(state_key, action)
        
        if next_state_key is None:
            # 终止状态
            target_q = reward
        else:
            # 非终止状态：使用贝尔曼方程
            next_state_value = self.get_state_value(next_state_key)
            target_q = reward + self.discount_factor * next_state_value
        
        # Q-learning更新
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_function[q_key] = new_q
        
        # 更新状态价值（使用最大Q值）
        # 这里简化处理，实际应该考虑所有可能的动作
        if state_key not in self.value_function:
            self.value_function[state_key] = new_q
        else:
            # 状态价值取最大Q值
            self.value_function[state_key] = max(
                self.value_function[state_key], 
                new_q
            )
    
    def calculate_reward(self, result: AttackResult, game: PlaneGame, 
                       player: int) -> float:
        """
        计算奖励
        
        Args:
            result: 攻击结果
            game: 游戏环境
            player: 玩家编号
            
        Returns:
            奖励值
        """
        if result == AttackResult.MISS:
            return -0.1  # 未击中小惩罚
        elif result == AttackResult.HIT:
            return 1.0   # 击中机身奖励
        elif result == AttackResult.DOWN:
            if game.get_winner() == player:
                return 100.0  # 获胜大奖励
            else:
                return 10.0   # 击落飞机奖励
        return 0.0
    
    def select_action(self, game: PlaneGame, player: int, 
                     training: bool = True) -> Tuple[int, int]:
        """
        选择动作（epsilon-greedy策略）
        
        Args:
            game: 游戏环境
            player: 玩家编号
            training: 是否在训练模式
            
        Returns:
            选择的动作 (row, col)
        """
        valid_actions = game.get_valid_actions(player)
        if not valid_actions:
            return None
        
        state = game.get_state(player)
        state_key = self.get_state_key(state['my_attack_record'])
        
        # epsilon-greedy策略
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            return valid_actions[np.random.randint(len(valid_actions))]
        else:
            # 利用：选择Q值最大的动作
            best_action = None
            best_q = float('-inf')
            
            for action in valid_actions:
                q_value = self.get_q_value(state_key, action)
                if q_value > best_q:
                    best_q = q_value
                    best_action = action
            
            # 如果所有动作Q值相同，随机选择
            if best_action is None:
                best_action = valid_actions[np.random.randint(len(valid_actions))]
            
            return best_action
    
    def learn_from_experience(self, state_key: str, action: Tuple[int, int],
                             reward: float, next_state_key: Optional[str]):
        """从经验中学习"""
        self.update_q_value(state_key, action, reward, next_state_key)
    
    def train_step(self, game: PlaneGame, player: int) -> Tuple[AttackResult, float]:
        """
        执行一步训练
        
        Args:
            game: 游戏环境
            player: 玩家编号
            
        Returns:
            (攻击结果, 奖励)
        """
        state = game.get_state(player)
        state_key = self.get_state_key(state['my_attack_record'])
        
        # 选择动作
        action = self.select_action(game, player, training=True)
        if action is None:
            return None, 0.0
        
        # 执行动作
        result = game.attack(player, action)
        reward = self.calculate_reward(result, game, player)
        
        # 获取下一状态
        next_state = game.get_state(player)
        next_state_key = self.get_state_key(next_state['my_attack_record'])
        
        # 如果游戏结束，下一状态为None
        if game.is_terminal():
            next_state_key = None
        
        # 学习
        self.learn_from_experience(state_key, action, reward, next_state_key)
        
        return result, reward
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'value_function': self.value_function,
            'q_function': self.q_function,
            'visit_count': self.visit_count,
            'board_size': self.board_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.value_function = model_data['value_function']
            self.q_function = model_data['q_function']
            self.visit_count = model_data.get('visit_count', {})
            self.board_size = model_data.get('board_size', 10)
            self.learning_rate = model_data.get('learning_rate', 0.1)
            self.discount_factor = model_data.get('discount_factor', 0.9)
            self.epsilon = model_data.get('epsilon', 0.1)
            print(f"模型已从 {filepath} 加载")
        else:
            print(f"模型文件不存在: {filepath}，使用新模型")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'num_states': len(self.value_function),
            'num_q_entries': len(self.q_function),
            'epsilon': self.epsilon
        }

