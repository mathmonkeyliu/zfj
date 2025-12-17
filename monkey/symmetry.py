from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config import GRID_SIZE


def _xy_to_a(x: int, y: int) -> int:
    return x * GRID_SIZE + y


def _a_to_xy(a: int) -> tuple[int, int]:
    return divmod(int(a), GRID_SIZE)


def _t_id(x: int, y: int) -> tuple[int, int]:
    return x, y


def _t_r90(x: int, y: int) -> tuple[int, int]:
    n = GRID_SIZE
    return y, n - 1 - x


def _t_r180(x: int, y: int) -> tuple[int, int]:
    n = GRID_SIZE
    return n - 1 - x, n - 1 - y


def _t_r270(x: int, y: int) -> tuple[int, int]:
    n = GRID_SIZE
    return n - 1 - y, x


def _t_mx(x: int, y: int) -> tuple[int, int]:
    """Mirror across vertical axis (flip columns)."""
    n = GRID_SIZE
    return x, n - 1 - y


def _t_my(x: int, y: int) -> tuple[int, int]:
    """Mirror across horizontal axis (flip rows)."""
    n = GRID_SIZE
    return n - 1 - x, y


def _t_d(x: int, y: int) -> tuple[int, int]:
    """Main diagonal reflection."""
    return y, x


def _t_ad(x: int, y: int) -> tuple[int, int]:
    """Anti-diagonal reflection."""
    n = GRID_SIZE
    return n - 1 - y, n - 1 - x


_TRANSFORMS_XY = (_t_id, _t_r90, _t_r180, _t_r270, _t_mx, _t_my, _t_d, _t_ad)


@dataclass(frozen=True, slots=True)
class SymmetryGroup:
    """
    D4 对称群：8 个变换。
    用 action id (0..99) 的映射表表示，便于快速检查“状态不变”的 stabilizer。
    """

    maps: tuple[tuple[int, ...], ...]  # 8 × 100

    @staticmethod
    def build() -> "SymmetryGroup":
        n = GRID_SIZE
        maps: list[tuple[int, ...]] = []
        for f in _TRANSFORMS_XY:
            m = [0] * (n * n)
            for a in range(n * n):
                x, y = _a_to_xy(a)
                x2, y2 = f(x, y)
                m[a] = _xy_to_a(x2, y2)
            maps.append(tuple(m))
        return SymmetryGroup(maps=tuple(maps))

    def apply(self, t_idx: int, a: int) -> int:
        return int(self.maps[int(t_idx)][int(a)])

    def stabilizer_transforms(self, shots: dict[int, int]) -> list[int]:
        """
        返回所有使“当前状态不变”的变换索引：
        对任意已观测格子 a，必须满足 result[a] == result[T(a)]。
        """
        if not shots:
            return list(range(len(self.maps)))

        stab: list[int] = []
        for ti, m in enumerate(self.maps):
            ok = True
            for a, v in shots.items():
                a2 = m[int(a)]
                v2 = shots.get(int(a2))
                if v2 is None or int(v2) != int(v):
                    ok = False
                    break
            if ok:
                stab.append(int(ti))
        return stab

    def canonical_action_under_stabilizer(self, a: int, stabilizer: Iterable[int]) -> int:
        """给定 stabilizer，返回 action 的轨道中最小的代表元。"""
        best = int(a)
        for ti in stabilizer:
            best = min(best, self.apply(int(ti), int(a)))
        return int(best)


