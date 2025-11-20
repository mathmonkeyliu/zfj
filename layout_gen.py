# layout_gen.py
import json
import numpy as np
from consts import *
import time

def get_plane_coords_set(head_x, head_y, direction):
    """获取单个飞机的坐标集合，用于碰撞检测"""
    coords = set()
    # 旋转逻辑
    rotations = [
        lambda x, y: (x, y),       # 0: Up
        lambda x, y: (-y, x),      # 1: Right
        lambda x, y: (-x, -y),     # 2: Down
        lambda x, y: (y, -x)       # 3: Left
    ]
    rot_func = rotations[direction]
    
    # 添加机头
    coords.add((head_x, head_y))
    
    # 添加机身部分
    for dx, dy in PLANE_SHAPE: # PLANE_SHAPE 来自 consts.py
        rx, ry = rot_func(dx, dy)
        nx, ny = head_x + rx, head_y + ry
        coords.add((nx, ny))
        
    return coords

def precompute_valid_planes():
    """预计算所有合法的单飞机位置"""
    valid_planes = [] # store list of coords
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            for d in range(4):
                coords = get_plane_coords_set(x, y, d)
                # 检查边界
                is_in_bounds = True
                for cx, cy in coords:
                    if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                        is_in_bounds = False
                        break
                
                if is_in_bounds:
                    # 存储格式：(coords_set, head_coord, full_list_for_json)
                    # head_coord 用于标记机头
                    valid_planes.append({
                        "coords": coords,
                        "head": (x, y)
                    })
    return valid_planes

def generate_strictly_unique():
    print("Step 1: 预计算单机合法位置...")
    all_planes = precompute_valid_planes()
    n = len(all_planes)
    print(f"单个飞机共有 {n} 种合法摆法。")
    
    print("Step 2: 遍历所有 3 机互不重叠组合 (C(n, 3))...")
    
    layouts = []
    total_attempts = 0
    
    # 核心回溯：为了去重，我们使用索引递增
    # i, j, k 代表 all_planes 列表中的索引
    for i in range(n):
        p1 = all_planes[i]
        s1 = p1["coords"]
        
        for j in range(i + 1, n):
            p2 = all_planes[j]
            s2 = p2["coords"]
            
            # 剪枝：如果有重叠，直接跳过
            if not s1.isdisjoint(s2):
                continue
                
            for k in range(j + 1, n):
                p3 = all_planes[k]
                s3 = p3["coords"]
                
                if s1.isdisjoint(s3) and s2.isdisjoint(s3):
                    # 找到合法解！
                    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
                    heads = []
                    
                    # 填充 Board
                    for plane in [p1, p2, p3]:
                        for (cx, cy) in plane["coords"]:
                            board[cy][cx] = 1 # 机身
                        hx, hy = plane["head"]
                        board[hy][hx] = 2 # 机头 (覆盖机身标记)
                        heads.append((hx, hy))
                    
                    layouts.append({
                        "board": board.tolist(),
                        "heads": heads
                    })
                
                total_attempts += 1
                
    print(f"生成完成！共找到 {len(layouts)} 种不重复布局。")
    
    print("Step 3: 保存到 jsonl...")
    with open("layouts.jsonl", "w") as f:
        for l in layouts:
            f.write(json.dumps(l) + "\n")
            
if __name__ == "__main__":
    t_start = time.time()
    generate_strictly_unique()
    print(f"总耗时: {time.time() - t_start:.2f} 秒")