# config.py
from enum import Enum, auto

GRID_SIZE = 10
LAYOUT_FILE = "layouts.jsonl"

# the relative coordinates of the body of the plane, (0, 0) is the head of the plane
RELATIVE_COORDS = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (0, -2), (-1, -3), (0, -3), (1, -3)]


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