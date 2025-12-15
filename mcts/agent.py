from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import GRID_SIZE, GridState
from environment import BombPlanesEnv, build_outcome_table
from mcts.config import MCTSConfig


@dataclass
class _BeliefNode:
    """
    POMCP belief node（对应一段 history）。
    - particles：该 history 下通过随机采样/传播收集到的 layout 索引（近似 belief）
    - action_nodes：action -> _ActionNode
    """

    particles: list[int]
    n: int
    action_nodes: dict[int, "_ActionNode"]


@dataclass
class _ActionNode:
    n: int
    w: float  # 累积回报（这里用 -steps_remaining 作为回报）
    obs_children: dict[int, _BeliefNode]  # obs(0/1/2) -> next belief node


def _ucb_score(parent_n: int, child: _ActionNode, c: float) -> float:
    if child.n <= 0:
        return float("inf")
    q = child.w / child.n
    return q + c * np.sqrt(np.log(parent_n + 1.0) / child.n)

def _best_action_elim_on_subset(
    outcomes: np.ndarray,
    label_ids: np.ndarray,
    subset_idx: np.ndarray,
    unshot: np.ndarray,
    *,
    action_candidates: np.ndarray | None = None,
) -> int:
    """
    在一个较小的 layout 子集（粒子集）上跑“排除法（minimax）”，作为 rollout/启发式。
    subset_idx: layout indices (1d)
    unshot: bool(100,)
    """
    features = action_candidates if action_candidates is not None else np.flatnonzero(unshot)
    if features.size == 0:
        raise ValueError("No available actions in subset elim.")

    y = label_ids[subset_idx]
    uniq_labels, inv = np.unique(y, return_inverse=True)
    m = int(uniq_labels.size)
    if m <= 1:
        return int(features[0])

    best_a = int(features[0])
    best_worst = 1 << 30
    best_expected = float("inf")
    best_head_prob = -1.0

    inv64 = inv.astype(np.int64, copy=False)
    for a in features:
        col = outcomes[subset_idx, int(a)].astype(np.int64, copy=False)  # 0/1/2
        combo = col * m + inv64
        cont = np.bincount(combo, minlength=3 * m).reshape(3, m)
        outcome_counts = cont.sum(axis=1).astype(np.float64, copy=False)
        n = float(outcome_counts.sum())
        if n <= 0:
            continue

        uniq_per_outcome = (cont > 0).sum(axis=1).astype(np.float64, copy=False)
        worst = int(uniq_per_outcome.max())
        probs = outcome_counts / n
        expected = float((probs * uniq_per_outcome).sum())
        head_prob = float(probs[2])

        if (
            (worst < best_worst)
            or (worst == best_worst and expected < best_expected - 1e-12)
            or (worst == best_worst and np.isclose(expected, best_expected) and head_prob > best_head_prob + 1e-12)
            or (
                worst == best_worst
                and np.isclose(expected, best_expected)
                and np.isclose(head_prob, best_head_prob)
                and int(a) < best_a
            )
        ):
            best_worst = worst
            best_expected = expected
            best_head_prob = head_prob
            best_a = int(a)

    return best_a


@dataclass
class MCTSAgent:
    """
    MCTS (POMCP-style) agent.

    关键点（满足你的“必须随机取样”要求）：
    - 每次 simulation 都会从当前候选集合 cand_idx 中均匀随机采样一个 layout（粒子）。
    - 在 simulation/rollout 内部，该 layout 决定观测（outcomes[layout, action]），是确定的。
    - 估值是多次随机取样的平均，因此是“剩余步数期望”的 Monte Carlo 估计。
    """

    outcomes: np.ndarray  # (N,100) uint8 in {0,1,2}
    label_ids: np.ndarray  # (N,) int32 (head-config id)
    labels: list[tuple[tuple[int, int], ...]]  # id -> canonical heads
    cfg: MCTSConfig = MCTSConfig()

    @staticmethod
    def from_layouts(layouts: list[dict[str, Any]], cfg: MCTSConfig | None = None) -> "MCTSAgent":
        outcomes, label_ids, labels = build_outcome_table(layouts)
        return MCTSAgent(outcomes=outcomes, label_ids=label_ids, labels=labels, cfg=cfg or MCTSConfig())

    def _new_belief_node(self) -> _BeliefNode:
        return _BeliefNode(particles=[], n=0, action_nodes={})

    def _new_action_node(self) -> _ActionNode:
        return _ActionNode(n=0, w=0.0, obs_children={})

    def _progressive_widening_actions(self, node: _BeliefNode, unshot_actions: np.ndarray, rng: np.random.Generator) -> None:
        """
        逐步扩展：避免每个节点一次性考虑全部未打动作。
        每次调用最多新增 1 个动作，直到达到上限。
        """
        if len(node.action_nodes) >= self.cfg.progressive_widening_k:
            return

        # 从未探索过的动作里随机选一个加入（随机取样，避免固定偏置）
        all_actions = np.flatnonzero(unshot_actions)
        if all_actions.size == 0:
            return
        # 过滤掉已有的动作
        mask = np.ones(all_actions.shape[0], dtype=bool)
        for i, a in enumerate(all_actions):
            if int(a) in node.action_nodes:
                mask[i] = False
        candidates = all_actions[mask]
        if candidates.size == 0:
            return

        a = int(candidates[int(rng.integers(0, candidates.size))])
        node.action_nodes[a] = self._new_action_node()

    def _select_action(self, node: _BeliefNode, unshot_actions: np.ndarray, rng: np.random.Generator) -> int | None:
        if not unshot_actions.any():
            return None

        # progressive widening: 增加可选动作集合（随机加入）
        self._progressive_widening_actions(node, unshot_actions, rng)
        if not node.action_nodes:
            # 兜底：随机选一个未打动作
            choices = np.flatnonzero(unshot_actions)
            return int(choices[int(rng.integers(0, choices.size))])

        # UCB 选择
        parent_n = max(1, node.n)
        best_a = None
        best_s = -1e100
        for a, an in node.action_nodes.items():
            if not unshot_actions[a]:
                continue
            s = _ucb_score(parent_n, an, self.cfg.c_ucb)
            if s > best_s:
                best_s = s
                best_a = a
        return best_a

    def _rollout(
        self,
        layout_i: int,
        unshot_actions: np.ndarray,
        heads_hit: int,
        rng: np.random.Generator,
        depth_left: int,
        belief_particles: np.ndarray,
    ) -> int:
        """
        rollout policy：在“随机取样得到的粒子集”上用排除法做启发式（粒子集会在 rollout 内按观测被过滤）。

        随机性要求体现在：粒子集来源于随机采样（Monte Carlo），不是确定全集遍历。
        返回“还需要多少步”。
        """
        steps = 0
        if depth_left <= 0:
            return 0

        particles = belief_particles
        if particles.size == 0:
            return 0

        while depth_left > 0 and heads_hit < 3 and unshot_actions.any():
            # 提速：只在 top-k 动作里评估（按 HEAD 概率挑一批候选动作）
            feats = np.flatnonzero(unshot_actions)
            if feats.size == 0:
                break
            if feats.size > self.cfg.rollout_action_k:
                head_prob = (self.outcomes[particles][:, feats] == 2).mean(axis=0)
                topk_idx = np.argpartition(-head_prob, self.cfg.rollout_action_k - 1)[: self.cfg.rollout_action_k]
                cand_actions = feats[topk_idx]
            else:
                cand_actions = feats

            a = _best_action_elim_on_subset(
                self.outcomes,
                self.label_ids,
                particles,
                unshot_actions,
                action_candidates=cand_actions,
            )
            unshot_actions[a] = False
            obs = int(self.outcomes[layout_i, a])
            if obs == 2:
                heads_hit += 1
            # belief proxy 更新：按这一步观测过滤粒子集
            col = self.outcomes[particles, int(a)]
            particles = particles[col == obs]
            if particles.size == 0:
                # 退化：粒子坍塌时停止 rollout（该分支的信息不足以继续做合理估值）
                break
            steps += 1
            depth_left -= 1
        return steps

    def _simulate(
        self,
        node: _BeliefNode,
        layout_i: int,
        unshot_actions: np.ndarray,
        heads_hit: int,
        rng: np.random.Generator,
        depth_left: int,
        belief_particles: np.ndarray,
    ) -> int:
        """
        一次 POMCP simulation。
        返回“从当前状态到终止（或截断）还需要的步数”。
        """
        node.n += 1
        node.particles.append(int(layout_i))

        if depth_left <= 0 or heads_hit >= 3 or (not unshot_actions.any()):
            return 0

        # 选动作
        a = self._select_action(node, unshot_actions, rng)
        if a is None:
            return 0

        # 执行动作（对采样 layout 来说观测是确定的）
        unshot_actions[a] = False
        obs = int(self.outcomes[layout_i, a])  # 0/1/2
        if obs == 2:
            heads_hit2 = heads_hit + 1
        else:
            heads_hit2 = heads_hit

        # 获取/创建 action node
        an = node.action_nodes.get(a)
        if an is None:
            an = self._new_action_node()
            node.action_nodes[a] = an

        # 获取/创建观测子节点
        child = an.obs_children.get(obs)
        if child is None:
            child = self._new_belief_node()
            an.obs_children[obs] = child

        # belief 粒子更新：用观测过滤粒子（随机取样的近似 belief 更新）
        colp = self.outcomes[belief_particles, int(a)]
        belief2 = belief_particles[colp == obs]
        if belief2.size == 0:
            belief2 = belief_particles  # 退化：保持原粒子集（避免数值坍塌）

        # 递归模拟（树内） or rollout（如果 child 还不够“熟”也可以直接 rollout，但这里统一走递归+rollout）
        remaining = 0
        if depth_left - 1 <= 0:
            remaining = 0
        else:
            # 若子节点还没怎么访问过，直接 rollout 会更高效
            if child.n == 0:
                rollout_left = min(self.cfg.rollout_depth, depth_left - 1)
                # 注意：rollout 需要拷贝 unshot_actions（避免污染）
                remaining = self._rollout(layout_i, unshot_actions.copy(), heads_hit2, rng, rollout_left, belief2)
            else:
                remaining = self._simulate(child, layout_i, unshot_actions, heads_hit2, rng, depth_left - 1, belief2)

        total_steps = 1 + int(remaining)

        # 回传：我们把“越少步越好”转成 reward = -steps
        an.n += 1
        an.w += float(-total_steps)

        return total_steps

    def _best_action_from_root(self, root: _BeliefNode, unshot_actions: np.ndarray, rng: np.random.Generator) -> int:
        """
        选择根节点动作：用访问次数最大（更稳），平手再看均值回报（更少步）。
        """
        # 若根没扩展出来（例如 simulations 太少），就随机
        if not root.action_nodes:
            choices = np.flatnonzero(unshot_actions)
            return int(choices[int(rng.integers(0, choices.size))])

        best_a = None
        best_n = -1
        best_q = -1e100
        for a, an in root.action_nodes.items():
            if not unshot_actions[a]:
                continue
            if an.n <= 0:
                continue
            q = an.w / an.n  # 越大越好（因为是 -steps）
            if (an.n > best_n) or (an.n == best_n and q > best_q):
                best_n = an.n
                best_q = q
                best_a = a

        if best_a is None:
            choices = np.flatnonzero(unshot_actions)
            return int(choices[int(rng.integers(0, choices.size))])
        return int(best_a)

    def choose_action(self, cand_idx: np.ndarray, unshot_actions: np.ndarray, heads_hit: int) -> int:
        """
        对当前 belief（候选集合）运行 MCTS，返回一个动作 id (0..99)。
        """
        if not unshot_actions.any():
            raise ValueError("No available actions.")

        rng = np.random.default_rng(self.cfg.seed)
        root = self._new_belief_node()

        # 根节点：直接开放到所有未打动作（避免 progressive widening 在根上“随机抽少量动作”导致策略接近随机）
        for a in np.flatnonzero(unshot_actions):
            root.action_nodes[int(a)] = self._new_action_node()

        # 若候选只有一种机头label，则直接打剩余机头（与其他算法一致）
        possible_labels = np.unique(self.label_ids[cand_idx])
        if possible_labels.size == 1:
            heads = self.labels[int(possible_labels[0])]
            for hx, hy in heads:
                a = int(hx) * GRID_SIZE + int(hy)
                if unshot_actions[a]:
                    return int(a)

        # 运行 simulations：每次从当前候选集合“随机采样一个真实布局粒子”
        for _ in range(int(self.cfg.num_simulations)):
            layout_i = int(cand_idx[int(rng.integers(0, cand_idx.size))])
            # 额外采样一批“belief 粒子”来近似 belief 更新（随机取样）
            particle_n = min(int(self.cfg.particle_count), int(cand_idx.size))
            belief_particles = cand_idx[rng.integers(0, int(cand_idx.size), size=particle_n, dtype=np.int32)]
            self._simulate(
                root,
                layout_i=layout_i,
                unshot_actions=unshot_actions.copy(),
                heads_hit=int(heads_hit),
                rng=rng,
                depth_left=int(self.cfg.max_depth),
                belief_particles=belief_particles,
            )

        return self._best_action_from_root(root, unshot_actions, rng)

    def play_one(self, env: BombPlanesEnv, *, layout: dict[str, Any], max_steps: int = 500) -> int:
        """
        在 env 的真实布局下进行一局，返回总步数（用于 evaluate.py）。
        """
        _, info = env.reset(layout=layout)
        steps = 0
        heads_hit = 0

        unshot = info["action_mask"].copy()
        cand_idx = np.arange(self.outcomes.shape[0], dtype=np.int32)
        shot: set[tuple[int, int]] = set()

        def shoot_xy(x: int, y: int) -> GridState:
            nonlocal steps, heads_hit
            sr = env.step((x, y))
            steps += 1
            shot.add((x, y))
            if sr.info["result"] == GridState.HEAD:
                heads_hit += 1
            return sr.info["result"]

        while heads_hit < 3 and steps < max_steps:
            if not unshot.any():
                break

            a = self.choose_action(cand_idx=cand_idx, unshot_actions=unshot, heads_hit=heads_hit)
            x, y = divmod(int(a), GRID_SIZE)
            unshot[int(a)] = False

            result = shoot_xy(int(x), int(y))
            obs_v = 2 if result == GridState.HEAD else (1 if result == GridState.BODY else 0)

            # 精确过滤候选集合（真实观测后）
            col = self.outcomes[cand_idx, int(a)]
            cand_idx = cand_idx[col == obs_v]
            if cand_idx.size == 0:
                break

        return steps


