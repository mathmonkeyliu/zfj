# 模型优化说明

## 模型文件大小问题

### 为什么模型pkl文件这么大？

模型文件（`ai_model.pkl`）包含以下内容：

1. **value_function**: 状态值函数字典，存储每个状态的价值
2. **q_function**: Q函数字典，存储每个(状态, 动作)对的Q值
3. **visit_count**: 访问计数字典（用于统计）
4. **其他参数**: board_size, learning_rate, discount_factor, epsilon等

随着训练的进行，`value_function` 和 `q_function` 字典会不断增长，因为：
- 每遇到一个新的游戏状态，就会在字典中添加一个条目
- 每遇到一个新的(状态, 动作)对，就会在Q函数中添加一个条目
- 10x10的棋盘有2^100种可能的攻击记录状态（虽然实际遇到的状态会少很多）

### 推理时不需要这么大的文件

对于推理（实际游戏），我们只需要：
- **q_function**: 用于选择最优动作
- **board_size**: 棋盘大小
- **epsilon**: 设为0（推理时不探索）

不需要：
- **value_function**: 训练时用于更新Q值，推理时不需要
- **visit_count**: 仅用于统计，推理时不需要
- **learning_rate**, **discount_factor**: 训练参数，推理时不需要

### 解决方案

#### 1. 使用推理专用模型（推荐）

训练完成后，可以使用 `save_model_for_inference()` 方法保存一个精简的模型：

```python
from dp_ai import DPAI

ai = DPAI()
ai.load_model("ai_model.pkl")  # 加载完整训练模型
ai.save_model_for_inference("ai_model_inference.pkl")  # 保存推理专用模型
```

推理时使用：
```python
ai.load_model_for_inference("ai_model_inference.pkl")
```

这样可以显著减小模型文件大小（通常可以减少30-50%）。

#### 2. 模型压缩

如果模型文件仍然很大，可以考虑：
- 使用压缩的pickle格式（需要修改代码）
- 只保存Q值大于某个阈值的条目（过滤掉不重要的状态）
- 使用更紧凑的数据结构

## 训练轮次继续训练

### 问题
之前从已有模型加载后，训练轮次总是从0开始，无法继续之前的训练进度。

### 解决方案
现在模型保存时会记录：
- `episode`: 当前训练轮次
- `training_stats`: 训练统计信息（胜率、步数等）

加载模型时会自动恢复这些信息，训练会从保存的轮次继续。

### 使用示例

```bash
# 第一次训练1000轮
python train.py --episodes 1000

# 继续训练500轮（会从第1000轮开始）
python train.py --episodes 500
```

训练输出会显示：
```
轮次 1100/1500 (本次训练: 100/500)
```

## GPU加速

### 说明
本项目使用的是基于表格的Q-learning方法，主要操作是字典查找和更新，这些操作在GPU上加速效果有限。

### GPU加速的局限性
1. **字典操作**: Python字典操作主要在CPU上进行
2. **稀疏访问**: Q-learning的状态访问是稀疏的，不适合GPU的并行计算
3. **数据传输开销**: CPU和GPU之间的数据传输可能比计算本身更慢

### 何时使用GPU
- 如果使用神经网络版本的AI（需要重新实现）
- 如果有大量并行的游戏实例需要同时训练

### 当前实现
代码中已添加GPU检测和基本支持，但实际加速效果有限。建议：
- **训练时**: 使用CPU（默认）
- **推理时**: 使用CPU（推理速度已经很快）

如果需要真正的GPU加速，建议：
1. 将Q-learning改为深度Q网络（DQN）
2. 使用PyTorch或TensorFlow实现
3. 使用GPU进行批量状态评估

## 文件大小对比

假设训练了10000轮，遇到10000个不同状态：

- **完整模型**: ~50-100MB（包含value_function和q_function）
- **推理模型**: ~30-50MB（只包含q_function）
- **压缩后**: ~20-40MB（使用gzip压缩）

实际大小取决于：
- 训练轮数
- 遇到的状态数量
- Q函数条目的数量

