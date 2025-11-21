import json
import random
import matplotlib.pyplot as plt
import numpy as np
from .solver import BattleAI
from config import State

def load_all_layouts(file_path="all_layouts.jsonl"):
    layouts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            layouts.append(data)
    return layouts

def simulate_game(layout, all_layouts):
    steps = 0
    max_steps = 1000
    ai = BattleAI(all_layouts)

    while ai.heads_hit < 3 and steps < max_steps:
        move = ai.get_best_move(method='test')
        if move is None:
            break
        
        x, y = move
        if [x, y] in layout['heads']:
            result = State.HEAD
        elif [x, y] in layout['bodies']:
            result = State.BODY
        else:
            result = State.MISS
        
        ai.update_state(x, y, result)
        steps += 1
    
    return steps

def run_statistics(all_layouts, num_samples=100, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    print(f"Loading layouts...")
    print(f"Loaded {len(all_layouts)} layouts")
    
    if num_samples > len(all_layouts):
        print(f"Warning: sample size {num_samples} > total layouts {len(all_layouts)}, using all layouts")
        sampled_layouts = all_layouts
    else:
        sampled_layouts = random.sample(all_layouts, num_samples)
    
    print(f"Simulating {len(sampled_layouts)} games...")
    
    steps_list = []
    for i, layout in enumerate(sampled_layouts):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(sampled_layouts)}")
        
        steps = simulate_game(layout, all_layouts)
        steps_list.append(steps)
    
    steps_array = np.array(steps_list)
    mean_steps = np.mean(steps_array)
    median_steps = np.median(steps_array)
    std_steps = np.std(steps_array)
    min_steps = np.min(steps_array)
    max_steps = np.max(steps_array)
    
    print("\n=== Statistics ===")
    print(f"Total samples: {len(steps_list)}")
    print(f"Mean steps: {mean_steps:.2f}")
    print(f"Median steps: {median_steps:.2f}")
    print(f"Std deviation: {std_steps:.2f}")
    print(f"Min steps: {min_steps}")
    print(f"Max steps: {max_steps}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(steps_list, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(mean_steps, color='r', linestyle='--', label=f'Mean: {mean_steps:.2f}')
    axes[0, 0].axvline(median_steps, color='g', linestyle='--', label=f'Median: {median_steps:.2f}')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Steps Distribution Histogram')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    sorted_steps = np.sort(steps_list)
    cumulative = np.arange(1, len(sorted_steps) + 1) / len(sorted_steps)
    axes[0, 1].plot(sorted_steps, cumulative, linewidth=2)
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution Function (CDF)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].boxplot(steps_list, vert=True)
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Steps Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    stats_text = f"""
Statistics Summary

Samples: {len(steps_list)}
Mean: {mean_steps:.2f}
Median: {median_steps:.2f}
Std Dev: {std_steps:.2f}
Min: {min_steps}
Max: {max_steps}

Percentiles:
  25%: {np.percentile(steps_array, 25):.2f}
  50%: {np.percentile(steps_array, 50):.2f}
  75%: {np.percentile(steps_array, 75):.2f}
  90%: {np.percentile(steps_array, 90):.2f}
  95%: {np.percentile(steps_array, 95):.2f}
  99%: {np.percentile(steps_array, 99):.2f}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('decision_tree/statistics_results_test.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: decision_tree/statistics_results_test.png")
    plt.show()
    
    return steps_list, {
        'mean': mean_steps,
        'median': median_steps,
        'std': std_steps,
        'min': min_steps,
        'max': max_steps,
        'samples': len(steps_list)
    }

if __name__ == "__main__":
    num_samples = 100
    random_seed = 42

    all_layouts = load_all_layouts()

    steps_list, stats = run_statistics(all_layouts, num_samples=num_samples, random_seed=random_seed)

