"""
炸飞机游戏图形界面
支持人机对战和AI自我对弈
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from game_env import PlaneGame, CellState, AttackResult, Direction
from dp_ai import DPAI
import threading
import time


class PlaneGameGUI:
    """游戏GUI"""
    
    CELL_SIZE = 40
    BOARD_MARGIN = 20
    
    def __init__(self, root):
        self.root = root
        self.root.title("炸飞机游戏")
        self.root.geometry("1200x700")
        
        self.game = PlaneGame()
        self.ai = DPAI()
        self.ai_mode = "human_vs_ai"  # human_vs_ai, ai_vs_ai
        self.current_player = 1
        self.ai_thinking = False
        
        # 尝试加载AI模型（优先使用推理专用模型）
        import os
        inference_model = "ai_model_inference.pkl"
        full_model = "ai_model.pkl"
        
        if os.path.exists(inference_model):
            try:
                self.ai.load_model_for_inference(inference_model)
                print(f"已加载推理专用模型: {inference_model}")
            except:
                try:
                    self.ai.load_model(full_model)
                except:
                    pass
        elif os.path.exists(full_model):
            try:
                self.ai.load_model(full_model)
                # 推理时不需要探索
                self.ai.epsilon = 0.0
            except:
                pass
        
        self.setup_ui()
        self.new_game()
    
    def setup_ui(self):
        """设置UI"""
        # 主容器
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Button(control_frame, text="新游戏", command=self.new_game, 
                 width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="随机布局", command=self.random_layout, 
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # 游戏模式选择
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(mode_frame, text="模式:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="human_vs_ai")
        tk.Radiobutton(mode_frame, text="人机对战", variable=self.mode_var, 
                      value="human_vs_ai", command=self.change_mode).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="AI自对弈", variable=self.mode_var, 
                      value="ai_vs_ai", command=self.change_mode).pack(side=tk.LEFT)
        
        # 状态标签
        self.status_label = tk.Label(control_frame, text="游戏状态", 
                                     font=("Arial", 12, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # 棋盘容器
        board_container = tk.Frame(main_frame)
        board_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 左侧：我的棋盘（显示对方攻击我的记录）
        left_frame = tk.Frame(board_container)
        left_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(left_frame, text="我的棋盘（对方攻击记录）", 
                font=("Arial", 14, "bold")).pack()
        self.my_board_canvas = tk.Canvas(left_frame, 
                                        width=self.CELL_SIZE * 10 + self.BOARD_MARGIN * 2,
                                        height=self.CELL_SIZE * 10 + self.BOARD_MARGIN * 2 + 30)
        self.my_board_canvas.pack()
        
        # 右侧：攻击棋盘（显示我攻击对方的记录）
        right_frame = tk.Frame(board_container)
        right_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(right_frame, text="攻击棋盘（我的攻击记录）", 
                font=("Arial", 14, "bold")).pack()
        self.attack_board_canvas = tk.Canvas(right_frame,
                                            width=self.CELL_SIZE * 10 + self.BOARD_MARGIN * 2,
                                            height=self.CELL_SIZE * 10 + self.BOARD_MARGIN * 2 + 30)
        self.attack_board_canvas.pack()
        
        # 绑定点击事件
        self.attack_board_canvas.bind("<Button-1>", self.on_cell_click)
    
    def change_mode(self):
        """切换游戏模式"""
        self.ai_mode = self.mode_var.get()
        self.new_game()
    
    def new_game(self):
        """开始新游戏"""
        self.game.reset()
        self.current_player = 1
        
        # 随机布局
        self.game.place_planes_random(1)
        self.game.place_planes_random(2)
        
        self.update_display()
        self.update_status("游戏开始！玩家1先手")
        
        # 如果是AI自对弈，自动开始
        if self.ai_mode == "ai_vs_ai":
            self.root.after(500, self.ai_vs_ai_loop)
    
    def random_layout(self):
        """重新随机布局"""
        if not self.game.is_terminal():
            self.game.place_planes_random(1)
            self.game.place_planes_random(2)
            self.update_display()
            self.update_status("已重新随机布局")
    
    def draw_board(self, canvas, board_data, clickable=False):
        """绘制棋盘"""
        canvas.delete("all")
        
        # 绘制坐标标签
        for i in range(10):
            # 行标签（数字）
            canvas.create_text(self.BOARD_MARGIN - 10, 
                            self.BOARD_MARGIN + i * self.CELL_SIZE + self.CELL_SIZE // 2,
                            text=str(i), font=("Arial", 10))
            # 列标签（字母）
            canvas.create_text(self.BOARD_MARGIN + i * self.CELL_SIZE + self.CELL_SIZE // 2,
                            self.BOARD_MARGIN - 20,
                            text=chr(ord('A') + i), font=("Arial", 10))
        
        # 绘制格子
        for row in range(10):
            for col in range(10):
                x1 = self.BOARD_MARGIN + col * self.CELL_SIZE
                y1 = self.BOARD_MARGIN + row * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                
                cell_state = board_data[row, col]
                
                # 根据状态设置颜色
                if cell_state == CellState.EMPTY.value:
                    color = "lightgray"
                elif cell_state == CellState.HIT_EMPTY.value:
                    color = "gray"  # 未击中
                elif cell_state == CellState.HIT_BODY.value:
                    color = "green"  # 击中机身/机翼
                elif cell_state == CellState.HIT_HEAD.value:
                    color = "red"  # 击中机头（击落）
                elif cell_state == CellState.PLANE_HEAD.value:
                    color = "blue"  # 调试模式显示飞机
                elif cell_state == CellState.PLANE_BODY.value:
                    color = "lightblue"  # 调试模式显示飞机
                else:
                    color = "white"
                
                # 绘制矩形
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", width=1)
                
                # 如果可点击，添加标记
                if clickable and cell_state == CellState.EMPTY.value:
                    # 鼠标悬停效果可以在这里添加
                    pass
    
    def update_display(self):
        """更新显示"""
        # 更新我的棋盘（显示对方攻击我的记录）
        my_board = self.game.render_board(1, show_planes=False)
        self.draw_board(self.my_board_canvas, my_board, clickable=False)
        
        # 更新攻击棋盘（显示我攻击对方的记录）
        attack_board = self.game.render_attack_board(1)
        self.draw_board(self.attack_board_canvas, attack_board, clickable=True)
    
    def update_status(self, message: str):
        """更新状态标签"""
        self.status_label.config(text=message)
    
    def on_cell_click(self, event):
        """处理格子点击"""
        if self.ai_mode != "human_vs_ai":
            return
        
        if self.current_player != 1:
            return
        
        if self.game.is_terminal():
            return
        
        if self.ai_thinking:
            return
        
        # 计算点击的格子
        col = int((event.x - self.BOARD_MARGIN) / self.CELL_SIZE)
        row = int((event.y - self.BOARD_MARGIN) / self.CELL_SIZE)
        
        if 0 <= row < 10 and 0 <= col < 10:
            # 检查是否已经攻击过
            if self.game.attack_record1[row, col] != 0:
                self.update_status("该位置已经攻击过了！")
                return
            
            # 执行攻击
            result = self.game.attack(1, (row, col))
            
            # 显示结果
            if result == AttackResult.MISS:
                self.update_status("未击中！")
            elif result == AttackResult.HIT:
                self.update_status("击中机身/机翼！")
            elif result == AttackResult.DOWN:
                self.update_status("击落一架飞机！")
            
            self.update_display()
            
            # 检查游戏是否结束
            if self.game.is_terminal():
                if self.game.get_winner() == 1:
                    messagebox.showinfo("游戏结束", "恭喜！你获胜了！")
                else:
                    messagebox.showinfo("游戏结束", "你输了！")
                return
            
            # AI回合
            self.current_player = 2
            self.ai_thinking = True
            self.update_status("AI思考中...")
            self.root.after(500, self.ai_turn)
    
    def ai_turn(self):
        """AI回合"""
        if self.game.is_terminal():
            self.ai_thinking = False
            return
        
        # AI选择动作
        action = self.ai.select_action(self.game, 2, training=False)
        
        if action is None:
            self.ai_thinking = False
            return
        
        row, col = action
        
        # 执行攻击
        result = self.game.attack(2, action)
        
        # 显示结果
        pos_str = f"{chr(ord('A') + col)}{row}"
        if result == AttackResult.MISS:
            self.update_status(f"AI攻击 {pos_str}：未击中")
        elif result == AttackResult.HIT:
            self.update_status(f"AI攻击 {pos_str}：击中机身/机翼！")
        elif result == AttackResult.DOWN:
            self.update_status(f"AI攻击 {pos_str}：击落一架飞机！")
        
        self.update_display()
        
        # 检查游戏是否结束
        if self.game.is_terminal():
            if self.game.get_winner() == 2:
                messagebox.showinfo("游戏结束", "AI获胜！")
            else:
                messagebox.showinfo("游戏结束", "你获胜了！")
            self.ai_thinking = False
            return
        
        # 切换回玩家
        self.current_player = 1
        self.ai_thinking = False
        self.update_status("轮到你攻击了！")
    
    def ai_vs_ai_loop(self):
        """AI自对弈循环"""
        if self.game.is_terminal():
            winner = self.game.get_winner()
            if winner:
                messagebox.showinfo("游戏结束", f"玩家{winner}获胜！")
            return
        
        # 当前玩家行动
        current_ai = self.ai
        action = current_ai.select_action(self.game, self.current_player, training=False)
        
        if action is None:
            return
        
        row, col = action
        pos_str = f"{chr(ord('A') + col)}{row}"
        
        # 执行攻击
        result = self.game.attack(self.current_player, action)
        
        # 显示结果
        player_name = f"玩家{self.current_player}"
        if result == AttackResult.MISS:
            self.update_status(f"{player_name}攻击 {pos_str}：未击中")
        elif result == AttackResult.HIT:
            self.update_status(f"{player_name}攻击 {pos_str}：击中机身/机翼！")
        elif result == AttackResult.DOWN:
            self.update_status(f"{player_name}攻击 {pos_str}：击落一架飞机！")
        
        self.update_display()
        
        # 检查游戏是否结束
        if self.game.is_terminal():
            winner = self.game.get_winner()
            if winner:
                messagebox.showinfo("游戏结束", f"玩家{winner}获胜！")
            return
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        
        # 继续下一回合（延迟以便观察）
        self.root.after(500, self.ai_vs_ai_loop)


def main():
    """主函数"""
    root = tk.Tk()
    app = PlaneGameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

