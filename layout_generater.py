import json
import itertools

from config import GRID_SIZE, LAYOUT_FILE, RELATIVE_COORDS, Direction


def rotate_point(dx: int, dy: int, direction: Direction) -> tuple[int, int]:
    """
    Rotate an offset (dx, dy) where:
    - dx = row offset
    - dy = col offset
    under the given direction, using matrix coordinates (row down, col right).
    """
    if direction == Direction.UP:
        return dx, dy
    if direction == Direction.DOWN:
        return -dx, -dy
    if direction == Direction.LEFT:
        # 90 deg CCW: (row, col) -> (col, -row)
        return dy, -dx
    if direction == Direction.RIGHT:
        # 90 deg CW: (row, col) -> (-col, row)
        return -dy, dx
    raise ValueError(f"Unknown direction: {direction}")


def get_valid_single_planes():
    valid_planes = []
    for d, head_x, head_y in itertools.product(Direction, range(GRID_SIZE), range(GRID_SIZE)):
        body_points = []
        valid = True
        head_point = (head_x, head_y)
        for dx, dy in RELATIVE_COORDS:
            dx, dy = rotate_point(dx, dy, d)
            absolute_x, absolute_y = head_x + dx, head_y + dy
            if not (0 <= absolute_x < GRID_SIZE and 0 <= absolute_y < GRID_SIZE):
                valid = False
                break
            body_points.append((absolute_x, absolute_y))
        if valid:
            valid_planes.append({
                "h": head_point,
                "d": d,
                "b": body_points,
                "all": set(body_points + [head_point])
            })
    return valid_planes


def generate_all_layouts():
    """
    Generate layouts grouped by 3-head configuration.

    Output format (jsonl, one head-pattern per line):
      {
        "heads": [[x1,y1],[x2,y2],[x3,y3]],          # canonical sorted by (x,y)
        "dir_triples": [["UP","LEFT","DOWN"], ...]  # each is directions aligned with heads order
      }

    Notes:
    - We store only direction triples to avoid repeating bodies in the file.
      Bodies can be derived from (heads + directions + RELATIVE_COORDS).
    """
    singles = get_valid_single_planes()
    groups: dict[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], list[list[str]]] = {}

    total_layouts = 0
    for p1, p2, p3 in itertools.combinations(singles, 3):
        if p1["all"] & p2["all"] or p1["all"] & p3["all"] or p2["all"] & p3["all"]:
            continue

        # Canonicalize within the group: order planes by head coordinate.
        planes = sorted([p1, p2, p3], key=lambda p: p["h"])
        heads = (planes[0]["h"], planes[1]["h"], planes[2]["h"])
        dirs = [str(planes[0]["d"]), str(planes[1]["d"]), str(planes[2]["d"])]
        groups.setdefault(heads, []).append(dirs)
        total_layouts += 1

    # Deterministic output order (nice for diffing/versioning).
    with open(LAYOUT_FILE, "w", encoding="utf-8") as f:
        for heads in sorted(groups.keys()):
            rec = {
                "heads": [[int(x), int(y)] for x, y in heads],
                "dir_triples": groups[heads],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated {total_layouts} valid layouts")
    print(f"Grouped into {len(groups)} head-pattern rows")
    print(f"Saved to {LAYOUT_FILE}")


if __name__ == "__main__":
    generate_all_layouts()
    