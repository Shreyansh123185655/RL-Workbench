"""
Advanced Visualizations for Comparative Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any


def plot_algorithm_comparison(results: Dict[str, Any], algorithms: List[str]) -> plt.Figure:
    """
    Create a comparison plot showing performance of multiple algorithms
    
    Args:
        results: Results dictionary from comparative analysis
        algorithms: List of algorithm names
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean rewards with error bars
    ax1 = axes[0, 0]
    means = [results[algo]['mean_reward'] for algo in algorithms if algo in results]
    stds = [results[algo]['std_reward'] for algo in algorithms if algo in results]
    valid_algos = [algo for algo in algorithms if algo in results]
    
    bars = ax1.bar(valid_algos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Average Reward Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Final rewards (last 100 episodes)
    ax2 = axes[0, 1]
    final_means = [results[algo]['final_reward_mean'] for algo in algorithms if algo in results]
    final_stds = [results[algo]['final_reward_std'] for algo in algorithms if algo in results]
    
    bars = ax2.bar(valid_algos, final_means, yerr=final_stds, capsize=5, alpha=0.7,
                   color=['#27ae60', '#2980b9', '#c0392b', '#d68910', '#8e44ad'])
    ax2.set_ylabel('Final Mean Reward', fontsize=12)
    ax2.set_title('Final Performance (Last 100 Episodes)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Training time comparison
    ax3 = axes[1, 0]
    times = [results[algo]['mean_time'] for algo in algorithms if algo in results]
    time_stds = [results[algo]['std_time'] for algo in algorithms if algo in results]
    
    bars = ax3.bar(valid_algos, times, yerr=time_stds, capsize=5, alpha=0.7,
                   color=['#1abc9c', '#16a085', '#e67e22', '#d35400', '#7f8c8d'])
    ax3.set_ylabel('Mean Training Time (s)', fontsize=12)
    ax3.set_title('Training Efficiency', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Episode length comparison
    ax4 = axes[1, 1]
    lengths = [results[algo]['mean_length'] for algo in algorithms if algo in results]
    length_stds = [results[algo]['std_length'] for algo in algorithms if algo in results]
    
    bars = ax4.bar(valid_algos, lengths, yerr=length_stds, capsize=5, alpha=0.7,
                   color=['#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#ecf0f1'])
    ax4.set_ylabel('Mean Episode Length', fontsize=12)
    ax4.set_title('Solution Efficiency', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_learning_curves_comparison(results: Dict[str, Any], algorithms: List[str]) -> plt.Figure:
    """
    Plot learning curves for all algorithms on the same plot
    
    Args:
        results: Results dictionary from comparative analysis
        algorithms: List of algorithm names
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, algo in enumerate(algorithms):
        if algo not in results:
            continue
        
        rewards = results[algo]['rewards']
        # Smooth the curve using moving average
        window = min(50, len(rewards) // 10)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = range(window, len(rewards) + 1)
        else:
            smoothed = rewards
            episodes = range(1, len(rewards) + 1)
        
        ax.plot(episodes, smoothed, label=algo, color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (Smoothed)', fontsize=12)
    ax.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_statistical_significance_heatmap(pairwise_tests: Dict[str, Any], algorithms: List[str]) -> plt.Figure:
    """
    Create a heatmap showing statistical significance between algorithm pairs
    
    Args:
        pairwise_tests: Dictionary of pairwise test results
        algorithms: List of algorithm names
    
    Returns:
        Matplotlib figure
    """
    n = len(algorithms)
    matrix = np.zeros((n, n))
    
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i == j:
                matrix[i, j] = 1.0  # Self-comparison
            else:
                key = f"{algo1}_vs_{algo2}"
                if key in pairwise_tests:
                    # Use p-value: lower = more significant
                    matrix[i, j] = pairwise_tests[key]['p_value']
                else:
                    matrix[i, j] = 1.0  # No data
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap (lower p-values = more significant = darker color)
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.set_yticklabels(algorithms, fontsize=10)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                text = ax.text(j, i, "-", ha="center", va="center", color="black", fontsize=12)
            else:
                key = f"{algorithms[i]}_vs_{algorithms[j]}"
                if key in pairwise_tests:
                    p_val = pairwise_tests[key]['p_value']
                    sig = "**" if pairwise_tests[key]['significant'] else ""
                    text = ax.text(j, i, f"{p_val:.3f}{sig}", ha="center", va="center", 
                                  color="white" if p_val < 0.05 else "black", fontsize=9)
    
    ax.set_title("Statistical Significance (p-values)\n** = Significant (p < 0.05)", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('p-value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig


def plot_convergence_comparison(results: Dict[str, Any], algorithms: List[str]) -> plt.Figure:
    """
    Plot convergence behavior comparison
    
    Args:
        results: Results dictionary from comparative analysis
        algorithms: List of algorithm names
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, algo in enumerate(algorithms):
        if algo not in results:
            continue
        
        rewards = results[algo]['rewards']
        # Calculate moving average with larger window for convergence detection
        window = min(100, len(rewards) // 20)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = range(window, len(rewards) + 1)
        else:
            smoothed = rewards
            episodes = range(1, len(rewards) + 1)
        
        # Calculate variance over time
        variance = []
        var_window = 50
        for i in range(len(smoothed)):
            start = max(0, i - var_window)
            if i - start > 0:
                variance.append(np.var(smoothed[start:i+1]))
            else:
                variance.append(0)
        
        ax2 = ax.twinx()
        line1 = ax.plot(episodes, smoothed, label=f'{algo} (Reward)', 
                       color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
        line2 = ax2.plot(episodes, variance, label=f'{algo} (Variance)', 
                        color=colors[idx % len(colors)], linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax.set_title('Convergence Analysis (Reward & Variance)', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig
