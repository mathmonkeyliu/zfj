# 炸飞机AI项目

本项目实现了一个可以自行学习的“炸飞机”AI，包含完整的强化学习训练脚本以及可交互的双面板图形化界面。AI可以与真人对弈，也可以在界面中自我对战。

## 飞机模型说明

每方拥有三架形状固定的飞机，共占 10 个格子。以机头坐标为 `(0, 0)`、朝上为例，其余格子为：

- 机翼：`(-2, -1)`、`(-1, -1)`、`(0, -1)`、`(1, -1)`、`(2, -1)`
- 桥梁：`(0, -2)`
- 尾翼：`(-1, -3)`、`(0, -3)`、`(1, -3)`

即示意如下：

```
WWWWW
  T
 TTT
  H
```

- `H`：机头（被击中即判定整架飞机击落）
- `W`：横向 5 格机翼
- `T`：桥梁（单格）与三格尾翼

飞机可朝上/下/左/右四个方向放置，且互不重叠。

## 主要特性

- ✅ `aircraft_ai` 包含棋盘、飞机、强化学习环境与DQN智能体的全部核心逻辑
- ✅ `train.py` 提供可配置的训练脚本，默认使用 DQN + 经验回放 + 目标网络
- ✅ `play_gui.py` 基于 Pygame，支持
  - 人类 vs AI，鼠标点选坐标
  - AI vs AI 自博弈（空格键切换）
  - 击中格子显示绿色、击落机头显示红色、未探测格子保持灰色
  - 左右两个面板分别展示“我方进攻”和“AI进攻”记录

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

## 强化学习训练

```bash
python train.py \
  --episodes 6000 \
  --batch-size 256 \
  --replay-size 80000 \
  --target-update 200 \
  --eval-interval 400
```

- 日志会输出每轮奖励、探索率、评估成绩
- 训练结束后模型保存在 `artifacts/aircraft_dqn.pt`
- 通过 `--device cuda` 可启用 GPU（若可用）

## 图形界面与对战

```bash
python play_gui.py --model artifacts/aircraft_dqn.pt
```

操作说明：

- 左侧面板：我方对 AI 的攻击记录（鼠标点击）
- 右侧面板：AI 对我方棋盘的攻击记录，同时用蓝色描边展示己方飞机形状
- 灰色：未攻击；深灰：未命中；绿色：机身/机翼命中；红色：机头击落
- `R` 重置新对局
- `Space` 切换 AI 自博弈（双方都由模型操控，便于观战或加速验证）

若未训练或未提供模型文件，界面会自动降级为随机策略，方便测试 UI。

## 项目结构

```
.
├── aircraft_ai/
│   ├── agent.py        # DQN网络、经验回放与策略封装
│   ├── board.py        # 棋盘与攻击判定
│   ├── constants.py    # 全局常量 & 奖励配置
│   ├── env.py          # AttackState 抽象与强化学习环境
│   ├── plane.py        # 飞机形状、旋转与坐标转换
│   └── __init__.py
├── play_gui.py         # Pygame 双面板可视化
├── train.py            # 训练脚本
├── requirements.txt
└── README.md
```

## 进一步扩展建议

- 在 `train.py` 中加入多智能体自博弈与 curriculum 训练，进一步提升命中效率
- 为 `play_gui.py` 加入自定义布阵或存档功能，使玩家可手动摆放飞机
- 引入模型评估脚本，比较不同超参数或奖励设计对性能的影响

欢迎根据需要继续优化策略、界面或奖励机制。祝玩得开心！
