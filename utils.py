import json
import numpy as np
from config import LAYOUT_FILE, GRID_SIZE, RELATIVE_COORDS, GridState, Direction


def rotate_point(dx: int, dy: int, direction: Direction) -> tuple[int, int]:
    if direction == Direction.UP:
        return -dx, -dy
    if direction == Direction.DOWN:
        return dx, dy
    if direction == Direction.LEFT:
        return dy, -dx
    if direction == Direction.RIGHT:
        return -dy, dx


def decode_layouts() -> dict[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], list[np.ndarray]]:
    result = {}
    with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            directions = data["d"]
            heads = tuple(tuple(head) for head in data["h"])
            result[heads] = []
            for d in directions:
                grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=int) * GridState.VOID
                for (hx, hy), direction in zip(heads, d):
                    grid[hx, hy] = GridState.HEAD
                    for dx, dy in RELATIVE_COORDS:
                        dx, dy = rotate_point(dx, dy, direction)
                        grid[hx + dx, hy + dy] = GridState.BODY
                grid = grid.flatten()
                result[heads].append(grid)
    return result