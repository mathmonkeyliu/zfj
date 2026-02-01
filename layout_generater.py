import json
from itertools import product, combinations

from config import GRID_SIZE, LAYOUT_FILE, RELATIVE_COORDS, Direction
from utils import rotate_point

def get_valid_single_planes():
    valid_planes = []
    for d, head_x, head_y in product(Direction, range(GRID_SIZE), range(GRID_SIZE)):
        body_points = []
        valid = True
        for dx, dy in RELATIVE_COORDS:
            dx, dy = rotate_point(dx, dy, d)
            absolute_x, absolute_y = head_x + dx, head_y + dy
            if not (0 <= absolute_x < GRID_SIZE and 0 <= absolute_y < GRID_SIZE):
                valid = False
                break
            body_points.append((absolute_x, absolute_y))
        if valid:
            valid_planes.append({
                "head": (head_x, head_y),
                "direction": d,
                "body": body_points,
                "all": set(body_points + [(head_x, head_y)])
            })
    return valid_planes


def generate_all_layouts():
    """
    Output format (jsonl, one head-pattern per line):
      {
        "h": ((x1,y1),(x2,y2),(x3,y3)), # canonical sorted by (x,y)
        "d": [(1,2,3), ...]             # the value is defined in config.Direction
      }
    """
    singles = get_valid_single_planes()
    groups = {}

    total_layouts = 0
    for p1, p2, p3 in combinations(singles, 3):
        if p1["all"] & p2["all"] or p1["all"] & p3["all"] or p2["all"] & p3["all"]:
            continue

        # Canonicalize within the group: order planes by head coordinate.
        planes = sorted([p1, p2, p3], key=lambda p: p["head"])
        heads = (planes[0]["head"], planes[1]["head"], planes[2]["head"])
        dirs = (planes[0]["direction"], planes[1]["direction"], planes[2]["direction"])
        groups.setdefault(heads, []).append(dirs)
        total_layouts += 1

    with open(LAYOUT_FILE, "w", encoding="utf-8") as f:
        for heads in sorted(groups.keys()):
            rec = {
                "h": heads,
                "d": groups[heads],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"There are {total_layouts} layouts in total")
    print(f"There are {len(groups)} handpiece-layouts in total")
    print(f"Saved to {LAYOUT_FILE}")


if __name__ == "__main__":
    generate_all_layouts()
    