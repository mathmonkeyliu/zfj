# id3.py
import numpy as np
from typing import List, Tuple

from config import GRID_SIZE, GridState
import utils


class ID3Agent:
    layouts: np.ndarray  # (66816, 100) uint8 in {0, 1, 2}
    head_indexs: np.ndarray # (66816,) int32 (head-pattern index)
    heads: List[Tuple[Tuple[int, int], ...]] # head-pattern index -> head-pattern
    symmetry_maps: List[np.ndarray] # symmetry maps

    def __init__(self):
        decode_layouts = utils.decode_layouts()
        layouts_list = []
        head_indexs_list = []
        heads_list = []
        for label_id, heads in enumerate(sorted(decode_layouts.keys())):
            heads_list.append(heads)
            layouts = decode_layouts[heads]
            for layout in layouts:
                # layout is (100,) with values 1, 2, 3. map to 0, 1, 2.
                layouts_list.append(layout - 1)
                head_indexs_list.append(label_id)       
        self.layouts = np.array(layouts_list, dtype=np.uint8)
        self.head_indexs = np.array(head_indexs_list, dtype=np.int32)
        self.heads = heads_list

        base = np.arange(GRID_SIZE * GRID_SIZE, dtype=np.int32).reshape(GRID_SIZE, GRID_SIZE)
        self.symmetry_maps = []
        transforms = []
        for i in range(4):
            transforms.append(np.rot90(base, i))
        for i in range(4):
            transforms.append(np.flipud(np.rot90(base, i)))
        for t in transforms:
            # argsort gives the inverse permutation (map[old_idx] = new_idx)
            self.symmetry_maps.append(np.argsort(t.flatten()))


    def select_move(self, observed: np.ndarray, topk: int = 1, possible_layout_indexs: np.ndarray = None) -> List[int]:
        """
        observed: (100,) array with GridState values (0=Unknown, 1=Void, 2=Body, 3=Head)
        possible_layout_indexs: Optional optimization. If provided, skip layout matching.
        """
        known_grid = (observed != GridState.UNKNOWN)
        unknown_grid = (observed == GridState.UNKNOWN)

        # If every cell is known, return []
        if not unknown_grid.any():
            return []
        
        if possible_layout_indexs is None:
            if not known_grid.any():    # If no cell is known, all layouts are possible
                possible_layout_indexs = np.arange(self.layouts.shape[0], dtype=np.int32)
            else:
                matches = np.all(self.layouts[:, known_grid] == observed[known_grid] - 1, axis=1)
                possible_layout_indexs = np.flatnonzero(matches).astype(np.int32)

        if possible_layout_indexs.size == 0:
            return []

        # If only 1 head pattern left, shoot the remaining heads
        possible_labels = np.unique(self.head_indexs[possible_layout_indexs])
        if possible_labels.size == 1:
            heads = self.heads[int(possible_labels[0])]
            targets = []
            for hx, hy in heads:
                idx = hx * GRID_SIZE + hy
                if observed[idx] == GridState.UNKNOWN:
                    targets.append(int(idx))
            
            if not targets:
                return []
            return targets[:topk]
        
        possible_layout_head_indexs = self.head_indexs[possible_layout_indexs]
        uniq_labels, inv = np.unique(possible_layout_head_indexs, return_inverse=True)
        uniq_labels_num = int(uniq_labels.size)

        unknown_grid_index = np.flatnonzero(unknown_grid)
        
        candidates = []
        
        for grid_index in unknown_grid_index:
            grid_state = self.layouts[possible_layout_indexs, grid_index].astype(np.int32) # shape: (possible_num, ) with values 0, 1, 2

            # grid_state & head_index
            combo = grid_state * uniq_labels_num + inv
            combo_count = np.bincount(combo, minlength=3 * uniq_labels_num).reshape(3, uniq_labels_num)
            state_counts = combo_count.sum(axis=1)
            n = int(state_counts.sum())
            
            if n == 0:
                continue
            
            entropy = 0.0
            for v in range(3):
                t = combo_count[v]
                if t.sum() == 0:
                    continue
                total = float(t.sum())
                p = t[t > 0].astype(np.float64, copy=False) / total
                entropy += (state_counts[v] / n) * -(p * np.log2(p)).sum()
            
            # Tie-breaking: prefer moves with higher probability of hitting HEAD
            head_prob = float(state_counts[2] / n)

            candidates.append((entropy, -head_prob, grid_index))
            
        candidates.sort()
        
        # Select top k with symmetry deduplication
        obs_grid = observed.reshape(GRID_SIZE, GRID_SIZE)
        valid_transforms = []
        for k in range(4):
            rot = np.rot90(obs_grid, k)
            if np.array_equal(obs_grid, rot):
                valid_transforms.append(self.symmetry_maps[k])
            flip = np.flipud(rot)
            if np.array_equal(obs_grid, flip):
                valid_transforms.append(self.symmetry_maps[k+4])
                
        selected = []
        seen = set()
        
        for _, _, action in candidates:
            if action in seen:
                continue
            selected.append(int(action))
            
            if len(selected) >= topk:
                break
            
            seen.add(action)
            for mapping in valid_transforms:
                seen.add(mapping[action])

        return selected
