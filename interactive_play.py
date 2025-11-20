import torch
import numpy as np
import os
import sys
import glob
from model import AlphaZeroNet
from mcts import AlphaZeroAgent
from consts import *
from config import MODEL_DIR

# --- 界面颜色设置 ---
class Colors:
    RESET = "\033[0m"
    GRAY = "\033[90m"      # Unknown / Miss
    GREEN = "\033[92m"     # Hit (Body)
    RED = "\033[91m"       # Kill (Head)
    YELLOW = "\033[93m"    # AI Prediction Highlight
    BLUE_BG = "\033[44m"   # Background for suggestions
    CYAN = "\033[96m"      # UI Elements
    BOLD = "\033[1m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(view_board, last_action=None, suggested_action=None):
    """
    view_board: 当前面板
    last_action: 上一步打的位置 (用于回顾)
    suggested_action: AI 本回合建议打的位置 (用于高亮显示)
    """
    print(f"\n  {Colors.CYAN}  0 1 2 3 4 5 6 7 8 9{Colors.RESET}")
    print(f"  {Colors.CYAN}┌─────────────────────┐{Colors.RESET}")
    
    for y in range(GRID_SIZE):
        line = f"{Colors.CYAN}{y} │{Colors.RESET} "
        for x in range(GRID_SIZE):
            idx = y * GRID_SIZE + x
            cell_val = view_board[y][x]
            
            char = "·"
            color = Colors.GRAY
            
            # 1. 确定基础字符和颜色
            if cell_val == STATE_MISS:
                char = "×"
                color = Colors.GRAY
            elif cell_val == STATE_HIT:
                char = "□"
                color = Colors.GREEN
            elif cell_val == STATE_KILL:
                char = "★"
                color = Colors.RED
            
            # 2. 覆盖显示逻辑
            
            # 情况 A: AI 建议攻击这个点 (通常这个点是 Unknown)
            if suggested_action == idx:
                # 显示为黄色的靶心
                line += f"{Colors.YELLOW}{Colors.BOLD}◎{Colors.RESET} "
            
            # 情况 B: 这是上一步打过的点 (且不是当前建议点)
            elif last_action == idx:
                # 给一个背景高亮，表示这是刚刚发生的事情
                line += f"{Colors.BLUE_BG}{color}{char}{Colors.RESET} "
            
            # 情况 C: 普通点
            else:
                line += f"{color}{char}{Colors.RESET} "
                
        line += f"{Colors.CYAN}│{Colors.RESET}"
        print(line)
    print(f"  {Colors.CYAN}└─────────────────────┘{Colors.RESET}")

def load_latest_model():
    models = sorted(glob.glob(os.path.join(MODEL_DIR, "bombing_plane_v*.pth")))
    if not models:
        print(f"{Colors.RED}错误: 没有找到 .pth 模型文件。请先运行 train.py{Colors.RESET}")
        sys.exit(1)
    latest = models[-1]
    return os.path.join(MODEL_DIR, latest)

def main():
    # 1. 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = load_latest_model()
    
    model = AlphaZeroNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    agent = AlphaZeroAgent(model, device=device)
    current_board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    planes_killed = 0
    steps = 0
    last_action = None
    
    # 2. 开始循环
    while planes_killed < 3:
        clear_screen()
        print(f"{Colors.YELLOW}{Colors.BOLD}=== 炸飞机 AI 助手 (v2.0 可视化增强版) ==={Colors.RESET}")
        print(f"模型: {os.path.basename(model_path)}")
        print(f"回合: {steps+1} | 已击落: {planes_killed}/3")
        
        # --- 关键修改：先计算，再画图 ---
        # AI 思考
        action = agent.select_action(current_board, epsilon=0.0)
        ax, ay = action % GRID_SIZE, action // GRID_SIZE
        
        # 画图 (传入 suggested_action)
        print_board(current_board, last_action=last_action, suggested_action=action)
        
        print(f"\nAI 建议攻击: {Colors.YELLOW}{Colors.BOLD}◎ ({ax}, {ay}){Colors.RESET}")
        print(f"{Colors.GRAY}图例: ◎=建议点, ×=未中, □=机身, ★=机头{Colors.RESET}")

        # 获取输入
        valid_input = False
        while not valid_input:
            res = input(f"\n请输入 ({ax},{ay}) 的结果 (m/h/k): ").strip().lower()
            
            if res in ['m', 'miss', '0']:
                current_board[ay][ax] = STATE_MISS
                valid_input = True
            elif res in ['h', 'hit', '1']:
                current_board[ay][ax] = STATE_HIT
                valid_input = True
            elif res in ['k', 'kill', '2', '3']:
                current_board[ay][ax] = STATE_KILL
                planes_killed += 1
                valid_input = True
            elif res == 'q':
                sys.exit(0)
        
        last_action = action
        steps += 1
        
        if (current_board == 0).sum() == 0 and planes_killed < 3:
            print("棋盘已满，游戏结束。")
            break

    clear_screen()
    print_board(current_board, last_action)
    print(f"\n{Colors.YELLOW}{Colors.BOLD}胜利！所有飞机已被击落。总步数: {steps}{Colors.RESET}")

if __name__ == "__main__":
    main()