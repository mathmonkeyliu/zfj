from __future__ import annotations

import torch
import torch.nn as nn

from .constants import BOARD_SIZE


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """AlphaZero风格的策略-价值网络"""
    
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_res_blocks: int = 4,
        num_channels: int = 128,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.eval_mode_flag = False
        
        # 输入层：1通道（知识面板）
        self.conv_input = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        
        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头：输出动作概率分布
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size * board_size),
        )
        
        # 价值头：输出标量价值
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        前向传播
        
        Args:
            x: 输入状态 (batch, board_size^2) 或 (batch, 1, board_size, board_size)
        
        Returns:
            policy: 动作概率分布 (batch, board_size^2)
            value: 状态价值 (batch, 1)
        """
        # 确保输入是4D tensor
        if x.dim() == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, 1, self.board_size, self.board_size)
        
        # 卷积特征提取
        x = self.conv_input(x)
        
        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        policy = self.policy_head(x)
        policy = torch.softmax(policy, dim=1)  # 归一化为概率分布
        
        # 价值头
        value = self.value_head(x)
        
        return policy, value
    
    def eval_mode(self):
        """上下文管理器，用于评估模式"""
        return self
    
    def __enter__(self):
        self.eval_mode_flag = True
        self.eval()
        return self
    
    def __exit__(self, *args):
        self.eval_mode_flag = False
        self.train()

