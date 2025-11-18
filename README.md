# 炸飞机游戏 - 动态规划AI

一个基于动态规划算法的"炸飞机"游戏AI实现，支持人机对战和AI自我对弈。

## 游戏规则

- 10×10的网格棋盘
- 每方有3架飞机
- 飞机形状为"士"字形，占据9个格子（1个机头、5个机身、3个机翼）
- 飞机可以朝向上、下、左、右四个方向放置
- 玩家轮流攻击对方坐标
- 击中机头即击落整架飞机
- 先击落对方全部三架飞机者获胜

## 安装依赖

```bash
pip install -r requirements.txt
```

或者直接安装：

```bash
pip install numpy
```

## 使用方法

### 1. 训练AI

训练AI模型（AI自我对弈学习）：

```bash
python main.py --mode train --episodes 1000 --save-interval 100
```

使用自我对弈模式（两个AI共享同一个模型）：

```bash
python main.py --mode train --episodes 1000 --self-play
```

或者直接使用训练脚本：

```bash
python train.py --episodes 1000 --save-interval 100
```

### 2. 启动图形界面

```bash
python main.py --mode gui
```

或者直接运行：

```bash
python gui.py
```

### 3. 游戏界面说明

- **我的棋盘**（左侧）：显示对方攻击我的记录
  - 灰色：未击中
  - 绿色：击中机身/机翼
  - 红色：击中机头（击落）

- **攻击棋盘**（右侧）：显示我攻击对方的记录
  - 灰色：未攻击或未击中
  - 绿色：击中机身/机翼
  - 红色：击中机头（击落）

### 4. 游戏模式

- **人机对战**：玩家点击攻击棋盘进行攻击，AI自动响应
- **AI自对弈**：两个AI自动对弈，可以观察AI的策略

## 文件结构

```
zfj/
├── game_env.py      # 游戏环境定义（规则、状态、动作）
├── dp_ai.py         # 动态规划AI算法
├── train.py         # 训练脚本
├── gui.py           # 图形界面
├── main.py          # 主程序入口
├── requirements.txt # 依赖列表
└── README.md        # 说明文档
```

## 核心API说明

### game_env.py

主要的游戏环境类 `PlaneGame`：

- `reset()`: 重置游戏
- `place_planes_random(player)`: 随机放置玩家的飞机
- `attack(attacker, target_pos)`: 执行攻击
- `get_state(player)`: 获取当前游戏状态
- `get_valid_actions(player)`: 获取所有有效动作
- `is_terminal()`: 检查游戏是否结束

### dp_ai.py

动态规划AI类 `DPAI`：

- `select_action(game, player, training)`: 选择动作
- `train_step(game, player)`: 执行一步训练
- `save_model(filepath)`: 保存模型
- `load_model(filepath)`: 加载模型

## 算法说明

使用Q-learning算法（一种动态规划方法）：
- 维护状态-动作值函数 Q(s, a)
- 使用epsilon-greedy策略平衡探索和利用
- 通过自我对弈不断学习最优策略

## 注意事项

1. 首次运行需要训练AI模型，建议至少训练1000轮
2. 训练时间取决于硬件性能，可能需要几分钟到几十分钟
3. 模型会保存为 `ai_model.pkl`，下次运行会自动加载
4. 如果模型文件不存在，AI会使用随机策略

