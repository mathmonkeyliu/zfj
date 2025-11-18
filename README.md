# 炸飞机AI项目

本项目实现了一个可以自行学习的"炸飞机"AI，使用 **AlphaZero** 算法进行训练。包含完整的训练脚本以及可交互的双面板图形化界面。AI可以与真人对弈，也可以在界面中自我对战。

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

- ✅ **AlphaZero 算法**：使用类似 AlphaZero 的算法，结合 MCTS 树搜索和策略-价值网络
- ✅ **残差网络**：使用 ResNet 架构提取深层特征
- ✅ **自我对弈训练**：AI 自己和自己下棋，生成高质量训练数据
- ✅ **图形化界面**：基于 Pygame 的双面板可视化，支持人机对战和 AI 自博弈

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

## 强化学习训练

使用 AlphaZero 风格训练：

```bash
python train_alphazero.py \
  --episodes 1000 \
  --self-play-games 100 \
  --mcts-simulations 800 \
  --c-puct 1.0 \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-3
```

**参数说明：**
- `--episodes`: 训练迭代次数
- `--self-play-games`: 每次迭代的自我对弈局数
- `--mcts-simulations`: MCTS 每次搜索的模拟次数（越多越强但越慢）
- `--c-puct`: PUCT 探索常数，控制探索与利用的平衡
- `--batch-size`: 训练批次大小
- `--epochs`: 每次迭代的训练轮数
- `--lr`: 学习率

**继续训练：**
```bash
python train_alphazero.py \
  --resume artifacts/alphazero.pt \
  --episodes 500
```

## 算法设计详解

### AlphaZero 算法

本项目实现了类似 **AlphaZero** 的算法，这是目前最强的棋类AI算法之一。

#### 核心组件

1. **策略-价值网络（Policy-Value Network）**
   - 使用残差卷积网络（ResNet）提取特征
   - **策略头**：输出动作概率分布 π(a|s)
   - **价值头**：输出状态价值 v(s) ∈ [-1, 1]
   - 网络架构：输入层 → 4个残差块 → 策略头/价值头

2. **蒙特卡洛树搜索（MCTS）**
   - **选择（Selection）**：使用 PUCT 公式选择动作
     - `U(s,a) = c_puct * P(s,a) * sqrt(ΣN(s,b)) / (1 + N(s,a))`
     - `a = argmax(Q(s,a) + U(s,a))`
   - **扩展（Expansion）**：遇到新状态时调用网络获取先验概率
   - **评估（Evaluation）**：网络输出价值 v(s)
   - **回溯（Backup）**：更新路径上所有节点的访问次数和累计价值
   - **走子（Play）**：根据访问次数计算改进的策略分布

3. **自我对弈训练循环**
   ```
   随机初始化网络 → 自我对弈生成数据 → 训练网络 → 评估性能 → 更新最佳网络 → 循环
   ```

4. **损失函数**（AlphaZero 标准损失）
   ```
   L = (z - v)^2 - π^T * log(p) + c||θ||^2
   ```
   - 价值损失：让网络预测的价值接近最终结果
   - 策略损失：让网络输出的策略接近 MCTS 改进的策略
   - L2 正则化：防止过拟合

#### 优势

- **无需人类知识**：完全从零开始学习
- **搜索与学习结合**：MCTS 利用网络指导搜索，搜索结果训练网络
- **高质量数据**：自我对弈生成的数据质量随训练不断提升
- **强大泛化能力**：同一套算法适用于多种游戏

#### 状态与动作

- **状态表示**：10×10 棋盘的知识面板，取值 {0: 未知, 1: 未中, 2: 击中机身, 3: 击中机头}，归一化到 [0,1]
- **动作空间**：100 个格子中尚未攻击的位置；`AttackState.action_mask()` 确保智能体只会选择未探索的坐标
- **奖励函数**：
  - 每步固定惩罚 `-0.05`，驱动策略尽快结束对局
  - 击中机头 +5 分，击落最后一架额外 +20
  - 命中机身不会获得奖励，只反映在状态中供后续推理

## 图形界面与对战

```bash
python play_gui.py --model artifacts/checkpoints/checkpoint_latest.pt --mcts-simulations 100
```

**参数说明：**
- `--model`: 模型路径（默认 `artifacts/alphazero.pt`）
- `--mcts-simulations`: MCTS 每次搜索的模拟次数（默认 100，可调低以加快速度）

**操作说明：**
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
│   ├── alphazero_net.py # AlphaZero 策略-价值网络（ResNet）
│   ├── board.py         # 棋盘与攻击判定
│   ├── constants.py     # 全局常量 & 奖励配置
│   ├── env.py           # AttackState 抽象与强化学习环境
│   ├── mcts.py          # 蒙特卡洛树搜索实现
│   ├── plane.py         # 飞机形状、旋转与坐标转换
│   └── __init__.py
├── train_alphazero.py   # AlphaZero 风格训练脚本
├── play_gui.py          # Pygame 双面板可视化
├── requirements.txt
└── README.md
```

## 进一步扩展建议

- 调整卷积层架构（如加入残差连接、注意力机制）以更好地捕捉长距离依赖
- 为 `play_gui.py` 加入自定义布阵或存档功能，使玩家可手动摆放飞机
- 引入模型评估脚本，比较不同超参数或奖励设计对性能的影响
- 实现课程学习（curriculum learning）：从简单布局逐步过渡到复杂布局
- 优化 MCTS 实现，支持更高效的状态复制和搜索

欢迎根据需要继续优化策略、界面或奖励机制。祝玩得开心！
