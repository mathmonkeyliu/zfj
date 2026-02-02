import argparse
import json
import time

from collections import defaultdict
import numpy as np
from typing import Dict, Tuple

from config import GRID_SIZE, GridState
from id3 import ID3Agent

PolicyType = Dict[Tuple[int, ...], Tuple[int, float]]

class MinAvgSolver:
    def __init__(self, topk: int, logging: bool = True):
        self.topk = topk
        self.logging = logging
        self.agent = ID3Agent()
        self.policy: PolicyType = {} # state_tuple -> (best_move, best_avg)
        self.visited_nodes = 0
        self.ref_counts = defaultdict(int)


    def prune(self, state_tuple: Tuple[int, ...]):
        # If this node is used by another parent, don't delete it
        if self.ref_counts[state_tuple] > 0:
            return
            
        if state_tuple not in self.policy:
            return
        
        # Get the move that was stored
        move, _ = self.policy[state_tuple]
        
        # Delete current node
        del self.policy[state_tuple]

        # Reconstruct children states and decrement their ref counts
        observed = np.array(state_tuple, dtype=np.int8)
        
        for outcome in [GridState.VOID, GridState.BODY, GridState.HEAD]:
            next_observed = observed.copy()
            next_observed[move] = outcome
            next_state_tuple = tuple(next_observed)
            
            self.ref_counts[next_state_tuple] -= 1
            self.prune(next_state_tuple)

    def solve(self, state_tuple: Tuple[int, ...], possible_indices: np.ndarray = None) -> float:
        self.visited_nodes += 1
        if self.logging and self.visited_nodes % 5000 == 0:
            print(f"Visited Nodes: {self.visited_nodes}, Policy Size: {len(self.policy)}")

        # 1. Check termination conditions
        observed = np.array(state_tuple, dtype=np.uint8)

        if possible_indices is None:
            known_grids = (observed != GridState.UNKNOWN)
            matches = np.all(self.agent.layouts[:, known_grids] == observed[known_grids] - 1, axis=1)
            possible_indices = np.flatnonzero(matches).astype(np.int32)
        possible_count = possible_indices.size

        if possible_count == 0:
            return 0.0

        # Check for unique head pattern
        current_head_labels = self.agent.head_indexs[possible_indices]
        unique_labels = np.unique(current_head_labels)

        if unique_labels.size == 1:
            label_idx = unique_labels[0]
            heads = self.agent.heads[label_idx]
            
            unhit_heads = 0
            for hx, hy in heads:
                idx = hx * GRID_SIZE + hy
                if observed[idx] == GridState.UNKNOWN:
                    unhit_heads += 1
            
            return float(unhit_heads)

        # 2. Check memoization
        if state_tuple in self.policy:
            _, avg = self.policy[state_tuple]
            return avg

        # 3. Select Top-K moves and recursively solve
        candidates = self.agent.select_move(observed, topk=self.topk, possible_layout_indexs=possible_indices)
        
        if not candidates:
            return 0.0

        best_avg = float('inf')
        best_move = -1
        
        move_results = []

        for move in candidates:
            move_values = self.agent.layouts[possible_indices, move]
            state_counts = np.bincount(move_values, minlength=3)
            
            total_steps = 0.0
            
            children_map = {} 

            for val, outcome_state_enum in [(0, GridState.VOID), (1, GridState.BODY), (2, GridState.HEAD)]:
                child_count = state_counts[val]
                if child_count == 0:
                    continue
                
                next_observed = observed.copy()
                next_observed[move] = outcome_state_enum
                next_state_tuple = tuple(next_observed)
                
                # Filter indices for child
                next_indices_mask = (move_values == val)
                next_possible_indices = possible_indices[next_indices_mask]
                
                child_avg = self.solve(next_state_tuple, next_possible_indices)

                total_steps += child_count * child_avg
                
                children_map[val] = next_state_tuple

            current_avg = 1.0 + (total_steps / possible_count)
            
            move_results.append({
                'avg': current_avg,
                'move': move,
                'children': children_map
            })

            if current_avg < best_avg:
                best_avg = current_avg
                best_move = move

        # 4. Save to Policy
        self.policy[state_tuple] = (best_move, best_avg)

        # 5. Prune non-optimal branches
        # Increment ref counts for the chosen best move's children
        for res in move_results:
            if res['move'] == best_move:
                for child_state in res['children'].values():
                    self.ref_counts[child_state] += 1
                break
        
        # Prune other branches
        for res in move_results:
            if res['move'] != best_move:
                for child_state in res['children'].values():
                    self.prune(child_state)

        return best_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output path for the checkpoint (.json)")
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    solver = MinAvgSolver(topk=args.topk)
    
    initial_state = tuple([GridState.UNKNOWN] * (GRID_SIZE * GRID_SIZE))

    t1 = time.time()
    avg_steps = solver.solve(initial_state)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")

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
