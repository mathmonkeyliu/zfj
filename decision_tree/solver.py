# solver.py
import json
import math
from config import LAYOUT_FILE, GRID_SIZE, State
from copy import deepcopy

KILL_WEIGHT = 0.13

class BattleAI:
    def __init__(self, layouts):
        self.layouts = layouts
        self.board_state = [[State.UNKNOWN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.valid_moves = [[x, y] for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        self.heads_hit = 0

    def update_state(self, x, y, result: State):
        self.board_state[x][y] = result
        if [x, y] in self.valid_moves:
            self.valid_moves.remove([x, y])
           
        if result == State.HEAD:
            self.heads_hit += 1
           
        remain_layouts = []
        target = [x, y]
       
        for layout in self.layouts:
            is_head = target in layout['heads']
            is_body = target in layout['bodies']
           
            match = False
            if result == State.MISS:
                if not is_head and not is_body: match = True
            elif result == State.BODY:
                if is_body: match = True
            elif result == State.HEAD:
                if is_head: match = True

            if match:
                remain_layouts.append(layout)
       
        self.layouts = remain_layouts

    def get_best_move(self, method: str = 'best'):
        if method == 'best':
            return self._maximum_entropy(regularization=KILL_WEIGHT)
        elif method == 'test':
            return self._test()
        elif method == 'minimum_entropy':
            return self._minimum_entropy()
        elif method == 'gini_index':
            return self._gini_index()
        else:
            raise ValueError(f"Invalid method: {method}")
        

    def _test(self, alpha: float = 1.5):
        pass


    def _minimum_entropy(self, regularization: float = 0.0):
        if self.heads_hit >= 3:
            return None
           
        if len(self.layouts) == 1:
            layout = self.layouts[0]
            remaining_heads = [h for h in layout['heads'] if self.board_state[h[0]][h[1]] != State.HEAD]
            if remaining_heads:
                return remaining_heads[0]
            return None

        # a little trick
        if len(self.valid_moves) == 100:
            return [3, 3]

        best_score = -float('inf')
        best_move = None
        total = len(self.layouts)
       
        for x, y in self.valid_moves:
            c_miss, c_body, c_head = 0, 0, 0
           
            for layout in self.layouts:
                if [x, y] in layout['heads']:
                    c_head += 1
                elif [x, y] in layout['bodies']:
                    c_body += 1
                else:
                    c_miss += 1
           
            entropy = 0
            for count in [c_miss, c_body, c_head]:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
           
            current_score = entropy + regularization * c_head / total
           
            if current_score > best_score:
                best_score = current_score
                best_move = [x, y]
               
        return best_move


    def _gini_index(self):
        if self.heads_hit >= 3:
            return None
           
        if len(self.layouts) == 1:
            layout = self.layouts[0]
            remaining_heads = [h for h in layout['heads'] if self.board_state[h[0]][h[1]] != State.HEAD]
            if remaining_heads:
                return remaining_heads[0]
            return None

        # a little trick
        if len(self.valid_moves) == 100:
            return [3, 3]

        best_score = 0
        best_move = None
        total = len(self.layouts)
       
        for x, y in self.valid_moves:
            c_miss, c_body, c_head = 0, 0, 0
           
            for layout in self.layouts:
                if [x, y] in layout['heads']:
                    c_head += 1
                elif [x, y] in layout['bodies']:
                    c_body += 1
                else:
                    c_miss += 1
           
            gini_index = 1
            for count in [c_miss, c_body, c_head]:
                if count > 0:
                    p = count / total
                    gini_index -= p * p
           
            current_score = gini_index
           
            if current_score > best_score:
                best_score = current_score
                best_move = [x, y]
               
        return best_move


    def dfs(self, layouts, board_state, valid_moves, heads_hit):
        if heads_hit == 3:
            return None, 0

        if len(layouts) == 1:
            layout = layouts[0]
            move = layout['heads'][0]
            step = len(layout['heads'])
            return move, step
            
        best_move = None
        best_steps = float('inf')

        for x, y in valid_moves:
            worst_case = 0
            for result in [State.MISS, State.BODY, State.HEAD]:
                new_heads_hit = heads_hit + (result == State.HEAD)
                new_layouts = []
                for layout in layouts:
                    is_head = [x, y] in layout['heads']
                    is_body = [x, y] in layout['bodies']
                    match = False
                    if result == State.MISS:
                        if not is_head and not is_body: match = True
                    elif result == State.BODY:
                        if is_body: match = True
                    elif result == State.HEAD:
                        if is_head: match = True
                    if match:
                        new_layouts.append(layout)
                if len(new_layouts) == 0:
                    continue
                if len(new_layouts) == len(layouts):
                    worst_case = float('inf')
                    break

                new_board_state = deepcopy(board_state)
                new_board_state[x][y] = result
                new_valid_moves = deepcopy(valid_moves)
                new_valid_moves.remove([x, y])

                _, remaining_steps = self.dfs(new_layouts, new_board_state, new_valid_moves, new_heads_hit)
                steps = 1 + remaining_steps
                print(f"Move: {x}, {y}, Result: {result}, Steps: {steps}")
                worst_case = max(worst_case, steps)

            if worst_case < best_steps:
                best_steps = worst_case
                best_move = [x, y]

        return best_move, best_steps


def load_layouts():
    layouts = []
    with open(LAYOUT_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            layouts.append(data)
    return layouts