import json
import itertools

from config import GRID_SIZE, LAYOUT_FILE, RELATIVE_COORDS, Direction


def rotate_point(x, y, direction):
    if direction == Direction.DOWN: return -x, -y
    if direction == Direction.LEFT: return y, -x
    if direction == Direction.UP: return x, y
    if direction == Direction.RIGHT: return -y, x


def get_valid_single_planes():
    valid_planes = []
    print("Calculating valid single plane positions...")
    
    for d in Direction:
        for head_x in range(GRID_SIZE):
            for head_y in range(GRID_SIZE):
                body_points = []
                valid = True
                head_point = (head_x, head_y)
                
                for relative_x, relative_y in RELATIVE_COORDS:
                    relative_x, relative_y = rotate_point(relative_x, relative_y, d)
                    absolute_x, absolute_y = head_x + relative_x, head_y + relative_y
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
    
    print(f"Valid single plane positions: {len(valid_planes)}")
    return valid_planes

def generate_all_layouts():
    singles = get_valid_single_planes()
    
    print("Generating three plane layouts, please wait...")

    count = 0
    with open(LAYOUT_FILE, 'w') as f:
        total_combos = itertools.combinations(singles, 3)
        
        for p1, p2, p3 in total_combos:
            if p1['all'] & p2['all'] or p1['all'] & p3['all'] or p2['all'] & p3['all']: continue
            layout = {
                "heads": [p1['h'], p2['h'], p3['h']],
                "directions": [str(p1['d']), str(p2['d']), str(p3['d'])],
                "bodies": list(p1['b']) + list(p2['b']) + list(p3['b'])
            }
            f.write(json.dumps(layout) + "\n")
            count += 1
                
    print(f"Generated {count} valid layouts")
    print(f"Saved to {LAYOUT_FILE}")

if __name__ == "__main__":
    generate_all_layouts()