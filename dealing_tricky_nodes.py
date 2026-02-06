import time
import os
import json
import multiprocessing
import shutil
import numpy as np
from typing import Tuple, Dict, List
from tqdm import tqdm

from min_avg_cpu import ParallelMinAvgSolver, _solve_subtree_wrapper
from config import GRID_SIZE, GridState
from min_avg import MinAvgSolver

SUB_TASK_MAX_NODES = 100

class SubtreeParallelSolver(ParallelMinAvgSolver):
    def __init__(self, topk):
        super().__init__(topk)
        # Reset policy for this subtree instance
        self.policy = {}

    def solve_specific_subtree(self, root_state, root_indices, task_id, temp_dir):
        # 1. Expand locally
        print(f"Expanding subtree for task {task_id} (limit {SUB_TASK_MAX_NODES})...")
        all_layers = []
        current_layer_list = [(root_state, root_indices)]
        
        while True:
            all_layers.append(current_layer_list)
            next_layer = self.get_layer_children(current_layer_list)
            
            if not next_layer:
                break
            
            # Stop if next layer is too big OR if it's getting too deep (heuristic)
            if len(next_layer) > SUB_TASK_MAX_NODES:
                all_layers.append(next_layer)
                break
            
            current_layer_list = next_layer
            
        frontier_layer = all_layers[-1]
        print(f"Task {task_id}: Expanded to {len(all_layers)} layers, frontier size {len(frontier_layer)}")

        # 2. Parallel execution of the frontier
        # Note: We do NOT write sub-files here, we keep in memory to aggregate into ONE json for this task_id
        task_list = []
        # Reuse logic from run_parallel but for this specific list
        # We need to handle the case where we are at the bottom (leafs) vs just stopped expansion
        # But get_layer_children handles logic. If it stopped because of limit, these are "unknown" subtrees to be solved.
        
        # Determine tasks
        # If we just stopped expansion, the nodes in frontier_layer need to be solved fully
        task_list = [(s, idx, self.topk) for s, idx in frontier_layer]
        
        print(f"Task {task_id}: Spawning {len(task_list)} parallel jobs...")
        
        with multiprocessing.Pool(processes=self.cpu_count) as pool:
            results_iter = pool.imap_unordered(_solve_subtree_wrapper, task_list)
            
            for state, avg, policy_frag in tqdm(results_iter, total=len(task_list), desc=f"Task {task_id} inner", leave=False):
                self.policy.update(policy_frag)

        # 3. Sequential Backpropagation for this subtree
        for i in range(len(all_layers) - 2, -1, -1):
            layer_nodes = all_layers[i]
            for s, idx in layer_nodes:
                self.solve(s, idx, prune_nodes=False)

        # 4. Global Pruning for this subtree
        print(f"Task {task_id}: Pruning unreachable nodes from subtree root...")
        self.prune_global(root_state)

        # 5. Save result
        print(f"Task {task_id}: Completed. Saving policy with {len(self.policy)} states.")
        
        data = {}
        for k, v in self.policy.items():
            key_str = "".join(map(str, k))
            data[key_str] = v
            
        filename = os.path.join(temp_dir, f"{''.join(map(str, root_state))}.json")
        with open(filename, "w") as f:
            json.dump(data, f)


def debug_missing_tasks(topk=2):
    print(f"Initializing solver with TopK={topk}...")
    # Just used for expansion logic
    solver = ParallelMinAvgSolver(topk=topk)
    initial_state = tuple([GridState.UNKNOWN] * (GRID_SIZE * GRID_SIZE))
    
    print(f"Re-running BFS expansion to identify frontier tasks...")
    
    # 1. Expand Layers (Re-using logic from min_avg_cpu.py to ensure same task order)
    all_layers = []
    current_layer_list = [(initial_state, None)]
    # Use same constant as min_avg_cpu.py to ensure indices match
    MAIN_EXPAND_NODES = 2000 
    
    while True:
        all_layers.append(current_layer_list)
        next_layer = solver.get_layer_children(current_layer_list)
        
        if not next_layer:
            break
        
        if len(next_layer) > MAIN_EXPAND_NODES:
            print(f"Frontier layer reached with {len(next_layer)} nodes.")
            all_layers.append(next_layer)
            break
            
        print(f"Layer expanded: {len(current_layer_list)} -> {len(next_layer)} nodes")
        current_layer_list = next_layer

    frontier_layer = all_layers[-1]
    temp_dir = os.path.join("temp_policies", f"topk_{topk}")
    
    if not os.path.exists(temp_dir):
        print(f"Warning: {temp_dir} does not exist. Cannot find existing progress.")
        os.makedirs(temp_dir, exist_ok=True)
        # We don't return here because we might want to start fresh or just see missing tasks
        
    # 2. Check existing files by state name
    existing_files = set(os.listdir(temp_dir))
    
    # 3. Find missing tasks
    missing_tasks = []
    for i, (s, idx) in enumerate(frontier_layer):
        state_str = "".join(map(str, s))
        fname = f"{state_str}.json"
        
        if fname not in existing_files:
            missing_tasks.append((i, s, idx))
    
    print(f"Found {len(missing_tasks)} missing tasks out of {len(frontier_layer)} total tasks.")
    
    if len(missing_tasks) == 0:
        print("No missing tasks found!")
        return

    # 4. Solve missing tasks in parallel
    print("Starting parallel execution for missing tasks...")
    print("-" * 50)

    for i, s, idx in missing_tasks:
        t0 = time.time()
        try:
            # Create a new solver instance for this subtree to have clean policy dict
            sub_solver = SubtreeParallelSolver(topk=topk)
            sub_solver.solve_specific_subtree(s, idx, i, temp_dir)
            
            duration = time.time() - t0
            # print(f"Task {i} completed in {duration:.2f}s")
        except Exception as e:
            print(f"\n!!! Error solving task {i}: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 50)
    print("All identified missing tasks processed.")

if __name__ == "__main__":
    debug_missing_tasks()
