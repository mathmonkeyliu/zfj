import argparse
import json
import time
import multiprocessing
import numpy as np
from typing import Tuple, Dict

from config import GRID_SIZE, GridState
from min_avg import MinAvgSolver

PolicyType = Dict[Tuple[int, ...], Tuple[int, float]]

def solve_subtree(state_tuple, possible_indices, topk, initial_policy=None):
    solver = MinAvgSolver(topk, logging=False)
    
    # Inject known results to avoid re-computation
    if initial_policy:
        solver.policy.update(initial_policy)
        
    avg = solver.solve(state_tuple, possible_indices)
    return state_tuple, avg, solver.policy

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

    def run_parallel(self, initial_state):
        print(f"Detected {self.cpu_count} logical CPU cores.")
        
        # 1. Expand and Store Layers
        all_layers = []
        current_layer_list = [(initial_state, None)]
        
        print(f"Starting BFS expansion with TopK={self.topk}...")
        
        while True:
            all_layers.append(current_layer_list)
            
            next_layer = self.get_layer_children(current_layer_list)
            
            # If next layer exceeds CPU count, stop expansion and use it as the parallel frontier
            if len(next_layer) > self.cpu_count:
                print(f"Next layer size {len(next_layer)} > cores {self.cpu_count}. Stopping expansion.")
                all_layers.append(next_layer)
                break
                
            print(f"Layer expanded: {len(current_layer_list)} -> {len(next_layer)} nodes")
            current_layer_list = next_layer

        # 2. Parallel Execution Phase
        t_start = time.time()
        for i in range(len(all_layers) - 1, -1, -1):
            t_start_layer = time.time()
            layer_nodes = all_layers[i]

            print(f"Processing Layer {i} ({len(layer_nodes)} nodes)...")

            # Determine tasks
            if i == len(all_layers) - 1:
                task_list = [(s, idx, self.topk) for s, idx in layer_nodes]
            else:
                children_cache = {}
                next_layer_nodes = all_layers[i+1]
                for child_state, _ in next_layer_nodes:
                    if child_state in self.policy:
                        children_cache[child_state] = self.policy[child_state]
                task_list = [(s, idx, self.topk, children_cache) for s, idx in layer_nodes]

            with multiprocessing.Pool(processes=self.cpu_count) as pool:
                # We use _solve_subtree_wrapper because imap only supports a single argument
                results_iter = pool.imap_unordered(_solve_subtree_wrapper, task_list)
                
                count = 0
                total = len(task_list)
                for state, avg, policy_frag in results_iter:
                    count += 1
                    print(f"Progress: Layer {i}, task {count}/{total} completed")
                    
                    # Merge results into main policy
                    self.policy.update(policy_frag)

            t_end_layer = time.time()
            print(f"Layer {i} completed in {t_end_layer - t_start_layer:.2f} seconds.")
        
        t_end = time.time()
        print(f"Parallel execution phase completed in {t_end - t_start:.2f} seconds.")
        return self.policy[initial_state][1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output path for the checkpoint (.json)")
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    t0 = time.time()

    solver = ParallelMinAvgSolver(topk=args.topk)
    initial_state = tuple([GridState.UNKNOWN] * (GRID_SIZE * GRID_SIZE))
    avg_steps = solver.run_parallel(initial_state)
    
    t1 = time.time()
    print(f"Total time taken: {t1 - t0:.2f} seconds")
    print(f"Solved. Min Avg Steps: {avg_steps:.3f}")

    checkpoint_data = {}
    for state, (move, _) in solver.policy.items():
        state_key = ",".join(map(str, state))
        checkpoint_data[state_key] = move

    print(f"Saving checkpoint to {args.out} with {len(checkpoint_data)} states...")
    with open(args.out, "w") as f:
        json.dump(checkpoint_data, f)
    print("Done.")

if __name__ == "__main__":
    main()
