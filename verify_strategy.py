import argparse
import json
import sys
import numpy as np
from collections import deque
from typing import Set, Tuple, Dict
from tqdm import tqdm

from config import GRID_SIZE, GridState
from id3 import ID3Agent

def parse_state_key(key: str) -> Tuple[int, ...]:
    return tuple(map(int, list(key)))

def state_to_key(state: Tuple[int, ...]) -> str:
    return "".join(map(str, state))

def check_reachability(strategy: Dict[str, int]) -> bool:
    print("Checking reachability...")
    
    # Root state is all UNKNOWN (0)
    initial_state = tuple([int(GridState.UNKNOWN)] * (GRID_SIZE * GRID_SIZE))
    initial_key = state_to_key(initial_state)
    
    visited_keys: Set[str] = set()
    queue = deque([initial_state])
    visited_keys.add(initial_key)
    
    # If the strategy doesn't even handle the root state (and it's not empty), that's a problem
    # unless the strategy is empty (trivial case).
    if not strategy:
        print("Strategy is empty.")
        return True

    if initial_key not in strategy:
        # It's possible the strategy is empty or immediate leaf? 
        # But if there are nodes in strategy, root must be there.
        print(f"Error: Root node {initial_key} not found in strategy, but strategy is not empty.")
        return False

    processed_count = 0
    while queue:
        current_state = queue.popleft()
        current_key = state_to_key(current_state)
        
        # If this state is a leaf in the strategy (not present), we stop expanding this branch
        if current_key not in strategy:
            continue
            
        move = strategy[current_key]
        processed_count += 1
        
        # Expand to 3 children: VOID, BODY, HEAD
        # Note: In strategy generation, children are only created if they are possible.
        # However, for reachability within the *strategy graph*, we should visit all children 
        # that are *defined* in the strategy. 
        # But the strategy dict doesn't store children explicitly, it implies them by state transitions.
        # So we generate the theoretical next states. If those keys exist in the strategy, we visit them.
        
        current_observed = list(current_state)
        
        # Possible outcomes: VOID(1), BODY(2), HEAD(3)
        for outcome in [GridState.VOID, GridState.BODY, GridState.HEAD]:
            next_observed = list(current_observed)
            # The move index gets the outcome value
            next_observed[move] = int(outcome)
            next_state = tuple(next_observed)
            next_key = state_to_key(next_state)
            
            # If the child state exists in the strategy, we add it to queue
            # We also need to track visited keys to find unreachable ones later
            if next_key in strategy:
                if next_key not in visited_keys:
                    visited_keys.add(next_key)
                    queue.append(next_state)
            
            # Note: A child might not be in strategy because it's a leaf node. 
            # That's fine. We only care about finding all *keys that are in the strategy*.

    # Check for unreachable nodes
    all_keys = set(strategy.keys())
    unreachable_keys = all_keys - visited_keys
    
    if unreachable_keys:
        print(f"Found {len(unreachable_keys)} unreachable (redundant) nodes.")
        # Print a few examples
        limit = 5
        print("Examples:")
        for k in list(unreachable_keys)[:limit]:
            print(k)
        return False
    
    print(f"Reachability check passed. Visited {len(visited_keys)} nodes.")
    return True

def verify_correctness(strategy: Dict[str, int]):
    print("Verifying correctness against layouts...")
    
    # Initialize Agent to load layouts
    try:
        agent = ID3Agent()
    except Exception as e:
        print(f"Failed to initialize ID3Agent (check layouts file): {e}")
        return False
        
    layouts = agent.layouts # (N, 100) values 0, 1, 2
    head_indexs = agent.head_indexs # (N,)
    
    num_layouts = layouts.shape[0]
    bad_layouts = 0
    
    # We iterate through every layout
    for idx in tqdm(range(num_layouts), desc="Verifying layouts"):
        current_layout = layouts[idx]
        
        # Start simulation
        current_state = [int(GridState.UNKNOWN)] * (GRID_SIZE * GRID_SIZE)
        
        steps = 0
        while True:
            state_key = "".join(map(str, current_state))
            
            if state_key not in strategy:
                # Reached a leaf node in the strategy.
                # Now check if this observation state uniquely determines the head arrangement.
                break
            
            move = strategy[state_key]
            
            # Get feedback from layout
            # layout values: 0, 1, 2. GridState: VOID=1, BODY=2, HEAD=3.
            val = current_layout[move]
            obs_val = val + 1 
            
            current_state[move] = obs_val
            steps += 1
            
            # Loop protection (though unlikely with finite states)
            if steps > 100: # Max steps = 100 cells
                print(f"Error: Simulation exceeded 100 steps for layout {idx}")
                return False

        # Verification Phase at Leaf
        # Filter all layouts that match the final observation
        # Optimization: We don't need to scan all layouts every time if we trust the logic,
        # but the prompt asks to "check current observe can uniquely determine".
        # So we verify against the ground truth of all layouts.
        
        final_obs = np.array(current_state, dtype=np.uint8)
        known_mask = (final_obs != GridState.UNKNOWN)
        
        # Get all layouts that match the known cells
        # agent.layouts is (N, 100), values 0..2
        # final_obs is 1..3
        # condition: layout[mask] == final_obs[mask] - 1
        
        if not known_mask.any():
            # If no observations, we match everything (bad strategy unless only 1 layout exists)
            matches = np.ones(num_layouts, dtype=bool)
        else:
            matches = np.all(agent.layouts[:, known_mask] == final_obs[known_mask] - 1, axis=1)
            
        matching_indices = np.flatnonzero(matches)
        
        # Check corresponding head indices
        matching_heads = head_indexs[matching_indices]
        unique_heads = np.unique(matching_heads)
        
        if unique_heads.size != 1:
            print(f"Bad Strategy! Layout {idx} led to ambiguity.")
            print(f"Final State: {state_key}")
            print(f"Matches {len(matching_indices)} layouts with {unique_heads.size} unique head patterns.")
            return False
            
    print(f"Verification passed. Checked {num_layouts} layouts.")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_strategy.py <strategy_file.json>")
        sys.exit(1)
        
    strategy_path = sys.argv[1]
    
    try:
        with open(strategy_path, 'r') as f:
            strategy = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {strategy_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {strategy_path}")
        sys.exit(1)
        
    print(f"Loaded strategy with {len(strategy)} nodes.")
    
    # Step 1: Reachability
    if not check_reachability(strategy):
        print("Strategy check failed: Unreachable nodes or invalid structure.")
        return
        
    # Step 2: Correctness
    if not verify_correctness(strategy):
        print("Strategy check failed: Incorrect behavior on some layouts.")
        return
        
    print("Strategy is VALID.")

if __name__ == "__main__":
    main()
