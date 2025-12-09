# interactive_play.py
import os
import time

from config import GRID_SIZE, GridState
from .solver import BattleAI, load_layouts

# --- 颜色代码 ---
C_RESET = "\033[0m"
C_GRAY = "\033[90m" # 未击中 (Miss)
C_GREEN = "\033[92m" # 机身 (Body)
C_RED = "\033[91m" # 机头 (Head)
C_YELLOW = "\033[93m" # 预测/提示
C_BLUE = "\033[94m" # 边框

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_board(ai_inst, suggestion=None):
    clear_screen()
    print(f"{C_BLUE}--- 炸飞机 AI 决策终端 ---{C_RESET}")
    print(f"{C_BLUE} " + " ".join([str(i) for i in range(GRID_SIZE)]) + f"{C_RESET}")
    print(f"{C_BLUE}  +" + "-" * (GRID_SIZE * 2 - 1) + f"+{C_RESET}")
   
    for y in range(GRID_SIZE):
        row_str = f"{C_BLUE}{y} |{C_RESET}"
        for x in range(GRID_SIZE):
            cell = ai_inst.board_state[x][y]
            symbol = "·"
            color = C_RESET
           
            if [x, y] == suggestion:
                symbol = "?"
                color = C_YELLOW
            elif cell == GridState.MISS:
                symbol = "■"
                color = C_GRAY # Miss
            elif cell == GridState.BODY:
                symbol = "X"
                color = C_GREEN # Body
            elif cell == GridState.HEAD:
                symbol = "H"
                color = C_RED # Head
           
            row_str += f"{color} {symbol}{C_RESET}"
       
        print(row_str)

def interactive_game():
    layouts = load_layouts()
   
    ai = BattleAI(layouts)
    print(f"{C_GREEN}模型载入完毕。当前可能性空间: {len(ai.layouts)} 种布局{C_RESET}")
   
    while True:
        draw_board(ai)
       
        if ai.heads_hit >= 3:
            print(f"\n{C_RED}恭喜！AI 已成功击落全部 3 架飞机！{C_RESET}")
            break
           
        if not ai.layouts:
            print(f"\n{C_RED}没有符合条件的布局！请检查您的输入是否正确！{C_RESET}")
            break
        print(f"\nAI 已击落机头数: {ai.heads_hit}/3. 剩余可能性: {C_RED}{len(ai.layouts)}{C_RESET}")
       
        start_time = time.time()
        move = ai.get_best_move()
        calc_time = time.time() - start_time
       
        draw_board(ai, suggestion=move)
       
        if move is None:
            print(f"{C_RED}AI 决策错误或游戏状态异常。{C_RESET}")
            break
        
        print(f"\nAI 已击落机头数: {ai.heads_hit}/3. 剩余可能性: {C_RED}{len(ai.layouts)}{C_RESET}")
        print(f"\nAI 建议打击坐标: {C_YELLOW}{move}{C_RESET} (计算耗时: {calc_time:.2f}s)")
        print("请告诉 AI 打击结果:")
        print(f" {C_GRAY}0 = 未击中 (Miss){C_RESET}")
        print(f" {C_GREEN}1 = 击中机身 (Body){C_RESET}")
        print(f" {C_RED}2 = 击中机头 (Head - 击落!){C_RESET}")
       
        res_str = input(f"{C_BLUE}输入结果 (0/1/2) > {C_RESET}")
        if res_str == '0':
            res = GridState.MISS
        elif res_str == '1':
            res = GridState.BODY
        elif res_str == '2':
            res = GridState.HEAD
        ai.update_state(move[0], move[1], res)
        print(f"收到反馈 ({move[0]},{move[1]}) -> {res}。剩余可能性: {C_RED}{len(ai.layouts)}{C_RESET}")

if __name__ == "__main__":
    interactive_game()