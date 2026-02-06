import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse the training log file and extract metrics."""
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract training metrics (LOSS values)
    train_pattern = r'\[Coach\] >>> TRAIN @Epoch: (\d+)\s+>>>\s+\|\| LOSS Avg: ([\d.]+)'
    train_data = re.findall(train_pattern, content)
    train_epochs = [int(e) for e, _ in train_data]
    train_losses = [float(l) for _, l in train_data]
    
    # Extract validation metrics
    valid_pattern = r'\[Coach\] >>> VALID @Epoch: (\d+)\s+>>>\s+\|\| RECALL@1 Avg: ([\d.]+) \|\| RECALL@10 Avg: ([\d.]+) \|\| RECALL@20 Avg: ([\d.]+) \|\| NDCG@10 Avg: ([\d.]+) \|\| NDCG@20 Avg: ([\d.]+)'
    valid_data = re.findall(valid_pattern, content)
    
    valid_epochs = [int(e) for e, _, _, _, _, _ in valid_data]
    recall_1 = [float(r) for _, r, _, _, _, _ in valid_data]
    recall_10 = [float(r) for _, _, r, _, _, _ in valid_data]
    recall_20 = [float(r) for _, _, _, r, _, _ in valid_data]
    ndcg_10 = [float(n) for _, _, _, _, n, _ in valid_data]
    ndcg_20 = [float(n) for _, _, _, _, _, n in valid_data]
    
    # Extract wall times for training
    wall_time_pattern = r'\[Wall TIME\] >>> ChiefCoach\.train takes ([\d.]+) seconds'
    wall_times = [float(t) for t in re.findall(wall_time_pattern, content)]
    
    # Extract best metric improvements
    best_pattern = r'\[Coach\] >>> Better \*\*\*NDCG@20\*\*\* of \*\*\*([\d.]+)\*\*\*'
    best_ndcg = [float(n) for n in re.findall(best_pattern, content)]
    
    return {
        'train_epochs': train_epochs,
        'train_losses': train_losses,
        'valid_epochs': valid_epochs,
        'recall_1': recall_1,
        'recall_10': recall_10,
        'recall_20': recall_20,
        'ndcg_10': ndcg_10,
        'ndcg_20': ndcg_20,
        'wall_times': wall_times,
        'best_ndcg': best_ndcg
    }

def visualize_metrics(metrics, output_path='log_visualization.png'):
    """Create comprehensive visualization of training metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Log Visualization - Amazon2014Electronics_550_MMRec', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(metrics['train_epochs'], metrics['train_losses'], marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Metrics (Recall)
    ax = axes[0, 1]
    ax.plot(metrics['valid_epochs'], metrics['recall_1'], marker='o', label='Recall@1', linewidth=2)
    ax.plot(metrics['valid_epochs'], metrics['recall_10'], marker='s', label='Recall@10', linewidth=2)
    ax.plot(metrics['valid_epochs'], metrics['recall_20'], marker='^', label='Recall@20', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall Score')
    ax.set_title('Validation Recall Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Metrics (NDCG)
    ax = axes[1, 0]
    ax.plot(metrics['valid_epochs'], metrics['ndcg_10'], marker='o', label='NDCG@10', linewidth=2)
    ax.plot(metrics['valid_epochs'], metrics['ndcg_20'], marker='s', label='NDCG@20', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NDCG Score')
    ax.set_title('Validation NDCG Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Time Per Epoch
    ax = axes[1, 1]
    epoch_range = range(len(metrics['wall_times']))
    colors = ['red' if t > 1000 else 'blue' for t in metrics['wall_times']]
    ax.bar(epoch_range, metrics['wall_times'], color=colors, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Wall Time Per Training Step (Red = Anomaly)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.show()

def print_summary(metrics):
    """Print a summary of key metrics."""
    print("\n" + "="*60)
    print("TRAINING LOG SUMMARY")
    print("="*60)
    
    print(f"\nTotal Training Epochs: {max(metrics['train_epochs']) + 1}")
    print(f"Validation Epochs: {len(metrics['valid_epochs'])}")
    print(f"Total Training Steps Logged: {len(metrics['wall_times'])}")
    
    print(f"\nTraining Loss:")
    print(f"  Initial: {metrics['train_losses'][0]:.5f}")
    print(f"  Final:   {metrics['train_losses'][-1]:.5f}")
    print(f"  Decrease: {metrics['train_losses'][0] - metrics['train_losses'][-1]:.5f}")
    
    print(f"\nValidation Metrics (Best):")
    print(f"  Recall@1:  {max(metrics['recall_1']):.4f}")
    print(f"  Recall@10: {max(metrics['recall_10']):.4f}")
    print(f"  Recall@20: {max(metrics['recall_20']):.4f}")
    print(f"  NDCG@10:   {max(metrics['ndcg_10']):.4f}")
    print(f"  NDCG@20:   {max(metrics['ndcg_20']):.4f} (Best metric)")
    
    print(f"\nWall Time Statistics:")
    print(f"  Average: {np.mean(metrics['wall_times']):.2f} seconds")
    print(f"  Min:     {min(metrics['wall_times']):.2f} seconds")
    print(f"  Max:     {max(metrics['wall_times']):.2f} seconds (Anomaly)")
    print(f"  Total:   {sum(metrics['wall_times'])/3600:.2f} hours")
    
    print(f"\nBest NDCG@20 Progression: {[f'{n:.4f}' for n in metrics['best_ndcg']]}")
    print("="*60 + "\n")

if __name__ == '__main__':
    log_file = Path('./logs/STAIR/Amazon2014Electronics_550_MMRec/0205150900/log.txt')
    
    if log_file.exists():
        print(f"Parsing {log_file}...")
        metrics = parse_log_file(log_file)
        
        print_summary(metrics)
        visualize_metrics(metrics)
    else:
        print(f"Error: Log file not found at {log_file}")
        print("Please ensure you're running this script from the STAIR project root directory.")
