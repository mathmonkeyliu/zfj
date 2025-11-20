# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from consts import GRID_SIZE, ACTION_SIZE

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        # 输入: 4 channels (Unknown, Miss, Hit, Kill) - One-hot 编码当前状态
        self.conv1 = nn.Conv2d(4, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块 (ResBlock)
        self.res1 = self._make_res_block(64)
        self.res2 = self._make_res_block(64)
        self.res3 = self._make_res_block(64)
        
        # Policy Head (输出 100 个格子的概率)
        self.p_conv = nn.Conv2d(64, 2, 1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * GRID_SIZE * GRID_SIZE, ACTION_SIZE)
        
        # Value Head (输出胜率/评估值)
        self.v_conv = nn.Conv2d(64, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # x shape: (batch, 10, 10) containing ints 0-3
        # 需要转为 One-hot: (batch, 4, 10, 10)
        x = F.one_hot(x.long(), num_classes=4).permute(0, 3, 1, 2).float()
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(x + self.res1(x))
        x = F.relu(x + self.res2(x))
        x = F.relu(x + self.res3(x))
        
        # Policy
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.p_fc(p)
        p = F.log_softmax(p, dim=1) # 输出 LogSoftmax
        
        # Value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return p, v