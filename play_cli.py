import torch
import numpy as np
import json
import random
import time
import os
from game import BattleGame
from model import AlphaZeroNet
from mcts import AlphaZeroAgent
from consts import *
from config import MODEL_DIR

# --- 颜色定义 (ANSI Escape Codes) ---
class Colors:
    RESET = "\033[0m"
    GRAY = "\033[90m"      # Unknown / Miss
    GREEN = "\033[92m"     # Hit (Body)
    RED = "\033[91m"       # Kill (Head)
    YELLOW = "\033[93m"    # Highlight AI choice
    CYAN = "\033[96m"      # Board Border

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(view_board, last_action=None, prob_map=None):
    """
    打印带颜色的棋盘
    view_board: 玩家看到的棋盘状态
    last_action: AI 上一步攻击的位置 (用于高亮)
    prob_map: (可选) AI 的概率热力图，用于显示 AI 在想什么
    """
    print(f"  {Colors.CYAN}   0 1 2 3 4 5 6 7 8 9{Colors.RESET}")
    print(f"  {Colors.CYAN} ┌─────────────────────┐{Colors.RESET}")
    
    for y in range(GRID_SIZE):
        line = f"{Colors.CYAN}{y}  │{Colors.RESET} "
        for x in range(GRID_SIZE):
            idx = y * GRID_SIZE + x
            cell_val = view_board[y][x]
            char = "."
            color = Colors.GRAY
            
            # 状态显示
            if cell_val == STATE_UNKNOWN:
                char = "·"
                # 如果有概率图，且该点未知，根据概率显示深浅（可选高级功能，这里简化）
            elif cell_val == STATE_MISS:
                char = "×" # Miss
                color = Colors.GRAY
            elif cell_val == STATE_HIT:
                char = "□" # Body
                color = Colors.GREEN
            elif cell_val == STATE_KILL:
                char = "★" # Head
                color = Colors.RED
            
            # 高亮 AI 最新的一步
            if last_action is not None and last_action == idx:
                # 给背景色或者加粗
                line += f"{Colors.YELLOW}{char}{Colors.RESET} "
            else:
                line += f"{color}{char}{Colors.RESET} "
                
        line += f"{Colors.CYAN}│{Colors.RESET}"
        
        # 在行末显示该行最大概率值（Debug AI 想法）
        if prob_map is not None:
            row_start = y * GRID_SIZE
            row_end = (y + 1) * GRID_SIZE
            max_p = np.max(prob_map[row_start:row_end])
            if max_p > 0.01:
                line += f"  {Colors.GRAY}max prob: {max_p:.3f}{Colors.RESET}"
                
        print(line)
        
    print(f"  {Colors.CYAN} └─────────────────────┘{Colors.RESET}")

def play_demo(model_path="bombing_plane_v50.pth", layout_file="layouts.jsonl"):
    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。请先运行 train.py。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    agent = AlphaZeroAgent(model, device=device)
    print("模型加载成功！")

    # 2. 加载随机布局
    print("正在抽取随机布局...")
    with open(layout_file, "r") as f:
        lines = f.readlines()
        if not lines:
            print("请先运行 layout_gen.py 生成布局！")
            return
        layout_data = json.loads(random.choice(lines))

    game = BattleGame(layout_data)
    steps = 0
    last_action = None
    prob_vis = None
    auto_mode = False

    while not game.done:
        clear_screen()
        print(f"{Colors.YELLOW}=== 炸飞机 AI 演示 (ResNet + MCTS) ==={Colors.RESET}")
        print(f"步数: {steps} | 剩余敌机: {game.planes_left}")
        print("-" * 30)
        
        # 获取 AI 思考概率 (仅用于展示)
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs, val = model(state_tensor)
            probs = torch.exp(log_probs).cpu().numpy()[0]
            valid = game.get_valid_moves()
            probs = probs * valid # 过滤非法步
            if probs.sum() > 0: probs /= probs.sum()
        
        print_board(state, last_action, probs)
        
        print("-" * 30)
        if last_action is not None:
            x, y = last_action % 10, last_action // 10
            res_str = ["未知", "未中", "击中机身!", "击落机头!!"][state[y][x]]
            print(f"AI 攻击了 ({x}, {y}) -> {res_str}")
            print(f"当前局面胜率预估 (V): {val.item():.4f}")

        if not auto_mode:
            cmd = input("\n按 [Enter] 继续，输入 [a] 开启自动，[q] 退出: ").strip().lower()
            if cmd == 'q': break
            if cmd == 'a': auto_mode = True
        else:
            time.sleep(0.3) # 自动模式下的延迟，方便观看

        # AI 行动
        # 这里使用 greedy 模式 (epsilon=0)，因为我们要展示最强水平
        action = agent.select_action(state, epsilon=0.0)
        _, _, done = game.step(action)
        
        last_action = action
        steps += 1

    # 游戏结束画面
    clear_screen()
    print(f"{Colors.RED}=== 游戏结束 ==={Colors.RESET}")
    print_board(game.get_state(), last_action)
    print(f"\n{Colors.YELLOW}最终成绩: {steps} 步清场！{Colors.RESET}")
    
    # 显示真实答案对比
    print("\n真实布局:")
    true_board = np.array(layout_data['board'])
    # 简单的转换显示
    print_board(true_board)

if __name__ == "__main__":
    # 注意：这里默认读取 v50 版本，你可以修改为你实际训练好的文件名
    # 如果你刚开始训练，可能没有 v50，请改成 v10 或你保存的最新模型
    import glob
    models = sorted(glob.glob(os.path.join(MODEL_DIR, "bombing_plane_v*.pth")))
    if models:
        latest_model = models[-1]
        play_demo(latest_model)
    else:
        print("未找到 .pth 模型文件，请先训练。")