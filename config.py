# config.py
from enum import Enum, auto

GRID_SIZE = 10
LAYOUT_FILE = "all_layouts.jsonl"

RELATIVE_COORDS = [
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
    (0, -2),
    (-1, -3), (0, -3), (1, -3)
]


class State(Enum):
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