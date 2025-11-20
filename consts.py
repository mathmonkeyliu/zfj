# consts.py

GRID_SIZE = 10
NUM_PLANES = 3

# 状态定义
STATE_UNKNOWN = 0
STATE_MISS = 1
STATE_HIT = 2  # 击中机身
STATE_KILL = 3 # 击中机头 (击落)

# 飞机的形状：相对于机头 (0,0) 的偏移量 (dx, dy)
# 描述：机头(0,0), 机身翼(-2,-1)到(2,-1), 连接处(0,-2), 机尾翼(-1,-3)到(1,-3)
# 注意：这里使用 (x, y) 坐标系。
PLANE_SHAPE = [
    (0, 0),   # Head
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), # Main Wing
    (0, -2),  # Body connector
    (-1, -3), (0, -3), (1, -3) # Tail Wing
]

# 动作数：100个格子
ACTION_SIZE = GRID_SIZE * GRID_SIZE