# config.py
from enum import Enum, auto

GRID_SIZE = 10
LAYOUT_FILE = "layouts.jsonl"

# Coordinate convention (IMPORTANT):
# A point is (x, y) where:
# - x = row index (0..9)
# - y = column index (0..9)
#
# Plane relative body coordinates (dx, dy) are expressed in (row_offset, col_offset),
# with (0, 0) being the head. For Direction.UP, the body is above the head (negative row offsets).
RELATIVE_COORDS = [
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (-2, 0),
    (-3, -1), (-3, 0), (-3, 1),
]


class GridState(Enum):
    UNKNOWN = auto()
    MISS = auto()
    BODY = auto()
    HEAD = auto()


class Direction(Enum):
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()
    def __str__(self):
        return self.name