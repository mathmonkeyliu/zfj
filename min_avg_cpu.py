import argparse
import json
import time
import multiprocessing
import numpy as np
import os
import shutil
from typing import Tuple, Dict
from tqdm import tqdm

from config import GRID_SIZE, GridState, MAX_EXPAND_NODES
from min_avg import MinAvgSolver

PolicyType = Dict[Tuple[int, ...], Tuple[int, float]]

def solve_subtree(state_tuple, possible_indices, topk, initial_policy=None):
    solver = MinAvgSolver(topk, logging=False)
    if initial_policy:
        solver.policy.update(initial_policy)
    avg = solver.solve(state_tuple, possible_indices)
    return state_tuple, avg, solver.policy

def solve_subtree_to_file(idx, temp_dir, state_tuple, possible_indices, topk):
    filename = os.path.join(temp_dir, f"{''.join(map(str, state_tuple))}.json")
    if os.path.exists(filename):
        return idx
    solver = MinAvgSolver(topk, logging=False)
    solver.solve(state_tuple, possible_indices)
    data = {}
    for k, v in solver.policy.items():
        key_str = "".join(map(str, k))
        data[key_str] = v
    with open(filename, "w") as f:
        json.dump(data, f)
    return idx

def _solve_subtree_to_file_wrapper(args):
    return solve_subtree_to_file(*args)

def _solve_subtree_wrapper(args):
    return solve_subtree(*args)

class ParallelMinAvgSolver(MinAvgSolver):
    def __init__(self, topk):
        super().__init__(topk, logging=True)
        self.cpu_count = multiprocessing.cpu_count()
        self.policy: PolicyType = {} # state_tuple -> (best_move, best_avg)
        
    def get_layer_children(self, layer_nodes):
        next_layer_map = {}
        
        for state, indices in layer_nodes:
            # Prepare context
            observed = np.array(state, dtype=np.uint8)
            
            if indices is None:
                known_grids = (observed != GridState.UNKNOWN)
                if not known_grids.any():
                    indices = np.arange(self.agent.layouts.shape[0], dtype=np.int32)
                else:
                    matches = np.all(self.agent.layouts[:, known_grids] == observed[known_grids] - 1, axis=1)
                    indices = np.flatnonzero(matches).astype(np.int32)
            
            possible_count = indices.size
            if possible_count == 0:
                continue

            # Check if Unique Head
            current_head_labels = self.agent.head_indexs[indices]
            if np.unique(current_head_labels).size == 1:
                continue
            
            # Select topk id3 moves
            candidates = self.agent.select_move(observed, topk=self.topk, possible_layout_indexs=indices)
            
            if not candidates:
                continue
            
            # expanding this node
            for move in candidates:
                move_values = self.agent.layouts[indices, move]
                state_counts = np.bincount(move_values, minlength=3)
                
                for val, outcome_state_enum in [(0, GridState.VOID), (1, GridState.BODY), (2, GridState.HEAD)]:
                    child_count = state_counts[val]
                    if child_count == 0:
                        continue
                    
                    # Create child
                    next_observed = observed.copy()
                    next_observed[move] = outcome_state_enum
                    next_state = tuple(next_observed)
                    
                    next_indices_mask = (move_values == val)
                    next_indices = indices[next_indices_mask]
                    
                    next_layer_map[next_state] = next_indices

        return list(next_layer_map.items())

    def run(self, initial_state):
        print(f"Detected {self.cpu_count} logical CPU cores.")
        
        # 1. Expand and Store Layers
        all_layers = []
        current_layer_list = [(initial_state, None)]
        
        print(f"Starting BFS expansion with TopK={self.topk}, Max Nodes={MAX_EXPAND_NODES}...")
        
        # BFS until MAX_EXPAND_NODES is reached
        while True:
            all_layers.append(current_layer_list)
            next_layer = self.get_layer_children(current_layer_list)
            if not next_layer: break
            print(f"Layer expanded: {len(current_layer_list)} -> {len(next_layer)} nodes")

            if len(next_layer) > MAX_EXPAND_NODES:
                print(f"Next layer size {len(next_layer)} > limit {MAX_EXPAND_NODES}. Stopping expansion.")
                all_layers.append(next_layer)
                break
            current_layer_list = next_layer

        # 2. Parallel Execution Phase (Last Layer Only)
        frontier_layer = all_layers[-1]
        print(f"Parallel processing frontier layer with {len(frontier_layer)} nodes...")
        
        temp_dir = os.path.join("temp_policies", f"topk_{self.topk}")
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Prepare tasks: (idx, temp_dir, state, indices, topk)
        task_list = [(i, temp_dir, s, idx, self.topk) for i, (s, idx) in enumerate(frontier_layer)]

        t_start = time.time()
        
        with multiprocessing.Pool(processes=self.cpu_count) as pool:
            results_iter = pool.imap_unordered(_solve_subtree_to_file_wrapper, task_list)
            for _ in tqdm(results_iter, total=len(task_list), desc="Frontier tasks"):
                pass
        
        # 3. Aggregate Results
        print("Aggregating partial policies from temp files...")
        
        # We iterate over the frontier layer to find expected files
        # Alternatively, we could just listdir, but iterating ensures we match the run structure
        # Actually, listing dir is safer if we want to pick up everything in the folder
        
        for fname in os.listdir(temp_dir):
            if not fname.endswith(".json"):
                continue
                
            path = os.path.join(temp_dir, fname)
            with open(path, "r") as f:
                data = json.load(f)
                # Convert keys back to tuples and update main policy
                for k_str, val in data.items():
                    # k_str is "123..."
                    k_tuple = tuple(map(int, list(k_str)))
                    self.policy[k_tuple] = (val[0], val[1])
        
        print(f"Aggregation complete. Policy size: {len(self.policy)}")
        
        # 4. Sequential Backpropagation (Upper Layers)
        # Iterate from the layer above frontier up to root
        for i in range(len(all_layers) - 2, -1, -1):
            layer_nodes = all_layers[i]
            print(f"Processing Layer {i} ({len(layer_nodes)} nodes) sequentially...")
            
            for s, idx in layer_nodes:
                # self.solve will use the cached children results in self.policy
                self.solve(s, idx, prune_nodes=False)

        # 5. Global Pruning
        print("Performing global pruning from root...")
        self.prune_global(initial_state)

        t_end = time.time()
        print(f"Execution completed in {t_end - t_start:.2f} seconds.")

        # Clean up temp
        # shutil.rmtree(temp_dir)
        
        if initial_state in self.policy:
            return self.policy[initial_state][1]
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output path for the checkpoint (.json)")
    parser.add_argument("--topk", type=int, required=True)
    args = parser.parse_args()

    t0 = time.time()

    solver = ParallelMinAvgSolver(topk=args.topk)
    initial_state = tuple[GridState, ...]([GridState.UNKNOWN] * (GRID_SIZE * GRID_SIZE))
    avg_steps = solver.run(initial_state)
    
    t1 = time.time()
    print(f"Total time taken: {t1 - t0:.2f} seconds")
    print(f"Solved. Min Avg Steps: {avg_steps:.3f}")

    checkpoint_data = {}
    for state, (move, _) in solver.policy.items():
        state_key = "".join(map(str, state))
        checkpoint_data[state_key] = move

    print(f"Saving checkpoint to {args.out} with {len(checkpoint_data)} states...")
    with open(args.out, "w") as f:
        json.dump(checkpoint_data, f)
    print("Done.")

if __name__ == "__main__":
    main()
