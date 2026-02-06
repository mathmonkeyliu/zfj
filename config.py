# config.py
from enum import IntEnum, unique
GRID_SIZE = 10
LAYOUT_FILE = "layouts.jsonl"

# min_avg config
MAX_EXPAND_NODES = 100

# Coordinate convention:
# A point is (x, y) where:
# - x = row index (0..9)
# - y = column index (0..9)
#
# Plane body's relative coordinates (dx, dy) are expressed in (row_offset, col_offset),
# with (0, 0) being the head. For Direction.UP, the head is above the body.
# UP is (-dx, -dy)
# DOWN is (dx, dy)
# LEFT is (dy, -dx)
# RIGHT is (-dy, dx)
RELATIVE_COORDS = [
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (-2, 0),
    (-3, -1), (-3, 0), (-3, 1),
]


@unique
class GridState(IntEnum):
    UNKNOWN = 0
    VOID = 1
    BODY = 2
    HEAD = 3


@unique
class Direction(IntEnum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
