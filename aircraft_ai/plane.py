from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, List, Sequence, Tuple

from .constants import BOARD_SIZE

Coordinate = Tuple[int, int]


class Orientation(Enum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()


class Segment(Enum):
    HEAD = auto()
    BODY = auto()
    WING = auto()


BASE_OFFSETS: Dict[Segment, Sequence[Coordinate]] = {
    Segment.HEAD: [(0, 0)],
    # 桥梁 + 尾翼：单个连接单元 + 底部 3 格
    Segment.BODY: [(0, -2), (-1, -3), (0, -3), (1, -3)],
    # 5 格水平机翼
    Segment.WING: [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)],
}


def rotate(point: Coordinate, orientation: Orientation) -> Coordinate:
    x, y = point
    if orientation == Orientation.UP:
        return x, y
    if orientation == Orientation.RIGHT:
        return y, -x
    if orientation == Orientation.DOWN:
        return -x, -y
    if orientation == Orientation.LEFT:
        return -y, x
    raise ValueError(orientation)


@dataclass(frozen=True)
class Plane:
    head: Coordinate
    orientation: Orientation

    def iter_cells(self) -> Iterable[Tuple[Coordinate, Segment]]:
        for segment, offsets in BASE_OFFSETS.items():
            for dx, dy in offsets:
                rx, ry = rotate((dx, dy), self.orientation)
                yield (self.head[0] + rx, self.head[1] + ry), segment

    def covered_cells(self) -> Dict[Coordinate, Segment]:
        return {coord: segment for coord, segment in self.iter_cells()}

    def within_bounds(self) -> bool:
        for (x, y), _ in self.iter_cells():
            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                return False
        return True

    @staticmethod
    def all_orientations() -> List[Orientation]:
        return [Orientation.UP, Orientation.RIGHT, Orientation.DOWN, Orientation.LEFT]
