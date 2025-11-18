"""
炸飞机游戏环境定义
包含游戏规则、状态管理、动作处理等核心API
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class CellState(Enum):
    """格子状态"""
    EMPTY = 0      # 空白
    PLANE_BODY = 1  # 飞机机身/机翼
    PLANE_HEAD = 2  # 飞机机头
    HIT_EMPTY = 3   # 已攻击但未击中（灰色）
    HIT_BODY = 4    # 已击中机身/机翼（绿色）
    HIT_HEAD = 5    # 已击中机头（红色）


class Direction(Enum):
    """飞机方向"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class AttackResult(Enum):
    """攻击结果"""
    MISS = 0      # 未击中
    HIT = 1       # 击中机身/机翼
    DOWN = 2      # 击落（击中机头）


class PlaneGame:
    """炸飞机游戏环境"""
    
    BOARD_SIZE = 10
    NUM_PLANES = 3
    
    # 飞机形状定义（以机头为(0,0)，方向为UP时的相对坐标）
    # 用户描述：机头(0,0)，其他格子(-2,-1), (-1,-1), (0,-1), (1,-1), (2,-1), (0,-2), (-1,-2), (1,-2)
    # 在屏幕坐标系中，y=-1和-2表示在机头上方，需要转换为row增加（向下）
    # 所以将y取反：y=-1 -> row+1, y=-2 -> row+2
    # 格式：(col_offset, row_offset)，其中row_offset是屏幕坐标系中的行偏移
    PLANE_SHAPE_UP = [
        (0, 0),      # 机头
        (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),  # 机身 part 1
        (0, 2),                                   # 机身 part 2
        (-1, 3), (0, 3), (1, 3)                   # 机身 part 3
    ]
    
    def __init__(self):
        """初始化游戏"""
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        # 玩家1的棋盘（自己的飞机布局）
        self.board1 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        # 玩家2的棋盘（对方的飞机布局）
        self.board2 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        
        # 攻击记录（玩家1攻击玩家2的记录）
        self.attack_record1 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        # 攻击记录（玩家2攻击玩家1的记录）
        self.attack_record2 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        
        # 飞机位置信息
        self.planes1 = []  # 玩家1的飞机列表 [(head_pos, direction), ...]
        self.planes2 = []  # 玩家2的飞机列表
        
        # 已击落的飞机数量
        self.down_planes1 = 0  # 玩家1被击落的飞机数
        self.down_planes2 = 0  # 玩家2被击落的飞机数
        
        # 游戏状态
        self.game_over = False
        self.winner = None
    
    def get_plane_cells(self, head_pos: Tuple[int, int], direction: Direction) -> List[Tuple[int, int]]:
        """
        获取飞机占据的所有格子坐标
        
        Args:
            head_pos: 机头位置 (row, col)
            direction: 飞机方向
            
        Returns:
            飞机占据的所有格子坐标列表
        """
        cells = []
        head_row, head_col = head_pos
        
        # 根据方向旋转坐标
        # PLANE_SHAPE_UP格式：(col_offset, row_offset)
        # UP: 机头在上，机身和机翼在下（row_offset为正）
        # DOWN: 机头在下，机身和机翼在上（row_offset取反）
        # LEFT: 机头在左，机身和机翼在右（交换col和row，row_offset取反）
        # RIGHT: 机头在右，机身和机翼在左（交换col和row，row_offset不变）
        if direction == Direction.UP:
            offsets = self.PLANE_SHAPE_UP
        elif direction == Direction.DOWN:
            # 180度旋转：col -> -col, row -> -row
            offsets = [(-col, -row) for col, row in self.PLANE_SHAPE_UP]
        elif direction == Direction.LEFT:
            # 逆时针90度：col -> row, row -> -col
            offsets = [(row, -col) for col, row in self.PLANE_SHAPE_UP]
        elif direction == Direction.RIGHT:
            # 顺时针90度：col -> -row, row -> col
            offsets = [(-row, col) for col, row in self.PLANE_SHAPE_UP]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        for dx, dy in offsets:
            row = head_row + dy
            col = head_col + dx
            cells.append((row, col))
        
        return cells
    
    def is_valid_placement(self, board: np.ndarray, head_pos: Tuple[int, int], 
                          direction: Direction) -> bool:
        """
        检查飞机位置是否有效（不重叠且在边界内）
        
        Args:
            board: 棋盘
            head_pos: 机头位置
            direction: 飞机方向
            
        Returns:
            是否有效
        """
        cells = self.get_plane_cells(head_pos, direction)
        
        # 检查是否所有格子都在边界内
        for row, col in cells:
            if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
                return False
        
        # 检查是否与已有飞机重叠
        for row, col in cells:
            if board[row, col] != CellState.EMPTY.value:
                return False
        
        return True
    
    def place_plane(self, board: np.ndarray, head_pos: Tuple[int, int], 
                   direction: Direction, plane_id: int) -> bool:
        """
        在棋盘上放置飞机
        
        Args:
            board: 棋盘
            head_pos: 机头位置
            direction: 飞机方向
            plane_id: 飞机ID（用于标记，1-3）
            
        Returns:
            是否成功放置
        """
        if not self.is_valid_placement(board, head_pos, direction):
            return False
        
        cells = self.get_plane_cells(head_pos, direction)
        head_row, head_col = head_pos
        
        for row, col in cells:
            if (row, col) == head_pos:
                board[row, col] = CellState.PLANE_HEAD.value
            else:
                board[row, col] = CellState.PLANE_BODY.value
        
        return True
    
    def place_planes_random(self, player: int):
        """
        随机放置玩家的所有飞机
        
        Args:
            player: 玩家编号（1或2）
        """
        board = self.board1 if player == 1 else self.board2
        planes = self.planes1 if player == 1 else self.planes2
        
        board.fill(CellState.EMPTY.value)
        planes.clear()
        
        import random
        attempts = 0
        max_attempts = 1000
        
        for plane_id in range(1, self.NUM_PLANES + 1):
            placed = False
            while not placed and attempts < max_attempts:
                attempts += 1
                head_row = random.randint(0, self.BOARD_SIZE - 1)
                head_col = random.randint(0, self.BOARD_SIZE - 1)
                direction = random.choice(list(Direction))
                
                if self.is_valid_placement(board, (head_row, head_col), direction):
                    if self.place_plane(board, (head_row, head_col), direction, plane_id):
                        planes.append(((head_row, head_col), direction))
                        placed = True
        
        if attempts >= max_attempts:
            raise RuntimeError(f"无法为玩家{player}放置所有飞机")
    
    def attack(self, attacker: int, target_pos: Tuple[int, int]) -> AttackResult:
        """
        执行攻击
        
        Args:
            attacker: 攻击者编号（1或2）
            target_pos: 攻击目标位置 (row, col)
            
        Returns:
            攻击结果
        """
        if self.game_over:
            return AttackResult.MISS
        
        row, col = target_pos
        
        # 确定目标和攻击记录
        if attacker == 1:
            target_board = self.board2
            attack_record = self.attack_record1
            planes = self.planes2
            down_count = self.down_planes2
        else:
            target_board = self.board1
            attack_record = self.attack_record2
            planes = self.planes1
            down_count = self.down_planes1
        
        # 检查是否已经攻击过
        if attack_record[row, col] != 0:
            return AttackResult.MISS  # 已经攻击过，无效攻击
        
        # 记录攻击
        cell_state = target_board[row, col]
        
        if cell_state == CellState.EMPTY.value:
            # 未击中
            attack_record[row, col] = CellState.HIT_EMPTY.value
            return AttackResult.MISS
        elif cell_state == CellState.PLANE_BODY.value:
            # 击中机身/机翼
            attack_record[row, col] = CellState.HIT_BODY.value
            return AttackResult.HIT
        elif cell_state == CellState.PLANE_HEAD.value:
            # 击中机头
            attack_record[row, col] = CellState.HIT_HEAD.value
            
            # 检查是否击落整架飞机（机头被击中即击落）
            if attacker == 1:
                self.down_planes2 += 1
                if self.down_planes2 >= self.NUM_PLANES:
                    self.game_over = True
                    self.winner = 1
            else:
                self.down_planes1 += 1
                if self.down_planes1 >= self.NUM_PLANES:
                    self.game_over = True
                    self.winner = 2
            
            return AttackResult.DOWN
        else:
            return AttackResult.MISS
    
    def get_state(self, player: int) -> Dict:
        """
        获取当前游戏状态（用于AI）
        
        Args:
            player: 玩家编号
            
        Returns:
            状态字典
        """
        if player == 1:
            my_attack_record = self.attack_record1
            opponent_attack_record = self.attack_record2
            my_down_count = self.down_planes1
            opponent_down_count = self.down_planes2
        else:
            my_attack_record = self.attack_record2
            opponent_attack_record = self.attack_record1
            my_down_count = self.down_planes2
            opponent_down_count = self.down_planes1
        
        return {
            'my_attack_record': my_attack_record.copy(),
            'opponent_attack_record': opponent_attack_record.copy(),
            'my_down_count': my_down_count,
            'opponent_down_count': opponent_down_count,
            'game_over': self.game_over,
            'winner': self.winner
        }
    
    def get_valid_actions(self, player: int) -> List[Tuple[int, int]]:
        """
        获取所有有效的攻击位置（未攻击过的位置）
        
        Args:
            player: 玩家编号
            
        Returns:
            有效动作列表
        """
        if player == 1:
            attack_record = self.attack_record1
        else:
            attack_record = self.attack_record2
        
        valid_actions = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if attack_record[row, col] == 0:
                    valid_actions.append((row, col))
        
        return valid_actions
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.game_over
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        return self.winner
    
    def render_board(self, player: int, show_planes: bool = False) -> np.ndarray:
        """
        渲染棋盘（用于显示）
        
        Args:
            player: 玩家编号
            show_planes: 是否显示飞机位置（用于调试）
            
        Returns:
            渲染后的棋盘
        """
        if player == 1:
            board = self.board1
            attack_record = self.attack_record2  # 对方攻击我的记录
        else:
            board = self.board2
            attack_record = self.attack_record1  # 对方攻击我的记录
        
        render = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                # 如果显示飞机位置（调试模式）
                if show_planes:
                    if board[row, col] == CellState.PLANE_HEAD.value:
                        render[row, col] = CellState.PLANE_HEAD.value
                    elif board[row, col] == CellState.PLANE_BODY.value:
                        render[row, col] = CellState.PLANE_BODY.value
                    else:
                        render[row, col] = CellState.EMPTY.value
                else:
                    # 正常模式：只显示攻击记录
                    if attack_record[row, col] != 0:
                        render[row, col] = attack_record[row, col]
                    else:
                        render[row, col] = CellState.EMPTY.value
        
        return render
    
    def render_attack_board(self, player: int) -> np.ndarray:
        """
        渲染攻击棋盘（显示我攻击对方的记录）
        
        Args:
            player: 玩家编号
            
        Returns:
            渲染后的攻击记录棋盘
        """
        if player == 1:
            attack_record = self.attack_record1
        else:
            attack_record = self.attack_record2
        
        return attack_record.copy()

