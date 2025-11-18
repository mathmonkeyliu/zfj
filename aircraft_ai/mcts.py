from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .constants import BOARD_SIZE


@dataclass
class MCTSNode:
    """MCTS树节点"""
    state: np.ndarray  # 当前状态
    action_mask: np.ndarray  # 合法动作mask
    parent: Optional[MCTSNode] = None
    action: Optional[int] = None  # 从父节点到达此节点的动作
    children: Dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0  # 访问次数 N(s,a)
    total_value: float = 0.0  # 累计价值总和
    prior_prob: float = 0.0  # 先验概率 P(s,a)
    is_terminal: bool = False  # 是否为终止状态
    
    @property
    def value(self) -> float:
        """平均价值 Q(s,a)"""
        return self.total_value / max(1, self.visit_count)
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children) == 0
    
    @property
    def is_expanded(self) -> bool:
        """是否已扩展"""
        if self.action_mask is None:
            return len(self.children) > 0 or self.is_terminal
        return len(self.children) > 0 or self.is_terminal or np.sum(self.action_mask) == 0


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(
        self,
        network: torch.nn.Module,
        step_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, float, bool]],
        c_puct: float = 1.0,
        num_simulations: int = 800,
        temperature: float = 1.0,
        device: torch.device = None,
    ):
        self.network = network
        self.step_fn = step_fn  # 函数：执行动作并返回 (next_state, reward, done)
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.device = device or torch.device("cpu")
    
    def search(self, state: np.ndarray, action_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        执行MCTS搜索，返回最佳动作和改进的策略分布
        
        Args:
            state: 当前状态 (board_size^2,)
            action_mask: 合法动作mask (board_size^2,)
        
        Returns:
            best_action: 最佳动作
            improved_policy: 改进的策略分布（基于访问次数）
        """
        root = MCTSNode(state=state.copy(), action_mask=action_mask.copy())
        
        # 检查是否为终止状态
        if np.sum(action_mask) == 0:
            root.is_terminal = True
            improved_policy = action_mask.copy()
            return 0, improved_policy
        
        # 扩展根节点
        if not root.is_expanded:
            self._expand_node(root)
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # 根据访问次数计算改进的策略
        improved_policy = self._get_improved_policy(root)
        
        # 选择最佳动作（训练时用温度采样，评估时用argmax）
        if self.temperature > 0:
            # 温度采样
            visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 
                                    for a in range(BOARD_SIZE * BOARD_SIZE)])
            visit_counts = visit_counts ** (1.0 / self.temperature)
            visit_counts = visit_counts * action_mask
            if visit_counts.sum() > 0:
                visit_counts = visit_counts / visit_counts.sum()
                best_action = int(np.random.choice(len(visit_counts), p=visit_counts))
            else:
                valid_actions = np.flatnonzero(action_mask)
                best_action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0
        else:
            # 选择访问次数最多的动作
            if root.children:
                best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            else:
                valid_actions = np.flatnonzero(action_mask)
                best_action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0
        
        return best_action, improved_policy
    
    def _simulate(self, root: MCTSNode) -> float:
        """
        从根节点执行一次模拟，返回价值估计
        
        Returns:
            value: 从当前节点视角的价值
        """
        node = root
        path = []
        
        # 选择阶段：从根节点到叶子节点
        while node.is_expanded and len(node.children) > 0 and not node.is_terminal:
            action = self._select_action(node)
            if action not in node.children:
                break
            node = node.children[action]
            path.append(node)
        
        # 扩展和评估阶段
        if node.is_terminal:
            # 游戏结束，返回实际结果
            value = self._evaluate_terminal(node)
        elif not node.is_expanded:
            # 扩展新节点
            value = self._expand_and_evaluate(node)
        else:
            # 已扩展但无子节点（不应该发生）
            value = 0.0
        
        # 回溯阶段：更新路径上所有节点的统计信息
        # 注意：在单人对战游戏中，价值不需要取反
        # 从叶子节点向根节点回溯
        current = node
        while current is not None and current is not root:
            current.visit_count += 1
            current.total_value += value
            current = current.parent
        
        # 更新根节点
        root.visit_count += 1
        root.total_value += value
        
        return value
    
    def _select_action(self, node: MCTSNode) -> int:
        """使用PUCT公式选择动作"""
        best_score = float('-inf')
        best_action = None
        
        if node.action_mask is None:
            # 如果action_mask为None，从children中选择
            if node.children:
                return max(node.children.items(), key=lambda x: x[1].visit_count)[0]
            return 0
        
        valid_actions = np.flatnonzero(node.action_mask)
        if len(valid_actions) == 0:
            return 0
        
        total_visits = sum(child.visit_count for child in node.children.values())
        
        for action in valid_actions:
            if action in node.children:
                child = node.children[action]
                # PUCT公式: Q + U
                q_value = child.value
                u_value = self.c_puct * child.prior_prob * np.sqrt(total_visits) / (1 + child.visit_count)
                score = q_value + u_value
            else:
                # 未访问的动作，需要获取先验概率
                # 这里简化处理，使用均匀分布（实际应该从网络获取）
                prior = 1.0 / len(valid_actions) if len(valid_actions) > 0 else 0.0
                u_value = self.c_puct * prior * np.sqrt(total_visits + 1)
                score = u_value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action is not None else valid_actions[0]
    
    def _expand_node(self, node: MCTSNode) -> None:
        """扩展节点（获取先验概率）"""
        if node.is_terminal:
            return
        
        # 检查action_mask
        if node.action_mask is None:
            # 如果action_mask为None，无法扩展
            return
        
        if np.sum(node.action_mask) == 0:
            node.is_terminal = True
            return
        
        # 如果node.state为None，使用父节点的状态
        state_to_use = node.state
        if state_to_use is None:
            if node.parent is not None:
                state_to_use = node.parent.state
            else:
                # 如果父节点也没有状态，使用零状态
                state_to_use = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        
        # 调用神经网络获取策略
        state_tensor = self._state_to_tensor(state_to_use)
        self.network.eval()
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
        
        policy = policy.squeeze().cpu().numpy()
        
        # 应用mask，归一化策略
        masked_policy = policy * node.action_mask
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            masked_policy = node.action_mask / node.action_mask.sum()
        
        # 创建子节点（但不立即执行动作，延迟到需要时）
        valid_actions = np.flatnonzero(node.action_mask)
        for action in valid_actions:
            child = MCTSNode(
                state=None,  # 延迟计算
                action_mask=None,  # 延迟计算
                parent=node,
                action=action,
                prior_prob=masked_policy[action],
            )
            node.children[action] = child
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """扩展节点并评估价值（执行动作获取新状态）"""
        if node.is_terminal:
            return self._evaluate_terminal(node)
        
        # 如果节点还未扩展，先扩展
        if len(node.children) == 0:
            self._expand_node(node)
        
        # 选择一个动作（使用先验概率采样）
        if node.action_mask is None:
            # 如果action_mask为None，从children中选择
            if node.children:
                action = max(node.children.items(), key=lambda x: x[1].prior_prob)[0]
            else:
                node.is_terminal = True
                return 0.0
        else:
            valid_actions = np.flatnonzero(node.action_mask)
            if len(valid_actions) == 0:
                node.is_terminal = True
                return 0.0
            
            # 使用先验概率采样动作
            if node.children:
                prior_probs = np.array([node.children[a].prior_prob if a in node.children else 1.0/len(valid_actions) 
                                       for a in valid_actions])
                prior_probs = prior_probs / prior_probs.sum()
                action = int(np.random.choice(valid_actions, p=prior_probs))
            else:
                action = int(np.random.choice(valid_actions))
        
        # 执行动作获取新状态
        # 注意：这里假设step_fn能够从node.state执行动作
        # 如果node.state是None，使用父节点的状态（简化处理）
        state_to_use = node.state if node.state is not None else (node.parent.state if node.parent else None)
        if state_to_use is None:
            # 如果无法获取状态，直接评估当前节点
            state_tensor = self._state_to_tensor(node.state if node.state is not None else np.zeros(BOARD_SIZE * BOARD_SIZE))
            self.network.eval()
            with torch.no_grad():
                _, value = self.network(state_tensor)
            return value.item()
        
        next_state, reward, done = self.step_fn(state_to_use, action)
        
        # 更新子节点状态
        if action in node.children:
            child = node.children[action]
            child.state = next_state
            # 计算新的action_mask（简化：假设所有未探索的格子都合法）
            child.action_mask = (next_state.reshape(BOARD_SIZE, BOARD_SIZE) == 0).astype(np.float32).reshape(-1)
            child.is_terminal = done
        
        # 评估价值
        if done:
            # 游戏结束，根据结果返回价值
            # 这里简化：如果获胜返回1，否则返回-1
            value = 1.0 if reward > 10.0 else -1.0
        else:
            # 调用神经网络评估价值
            next_state_tensor = self._state_to_tensor(next_state)
            self.network.eval()
            with torch.no_grad():
                _, value = self.network(next_state_tensor)
            value = value.item()
        
        return value
    
    def _evaluate_terminal(self, node: MCTSNode) -> float:
        """评估终止状态（游戏结束）"""
        # 这里需要根据实际游戏规则判断胜负
        # 简化处理：返回0（平局）
        return 0.0
    
    def _get_improved_policy(self, root: MCTSNode) -> np.ndarray:
        """根据访问次数计算改进的策略分布"""
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits > 0:
            for action, child in root.children.items():
                policy[action] = child.visit_count / total_visits
        
        # 应用mask并归一化
        policy = policy * root.action_mask
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = root.action_mask / root.action_mask.sum()
        
        return policy
    
    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """将状态转换为tensor"""
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device)
