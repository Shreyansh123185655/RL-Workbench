"""
Comparative Analysis Module
Runs multiple algorithms and performs statistical comparisons
"""

from typing import List, Callable, Tuple, Dict, Any
import numpy as np
from scipy import stats
import time


def run_comparative_analysis(
    state_space: List[int],
    actions: List[int],
    step: Callable,
    reset: Callable,
    algorithms: List[str],
    algorithm_functions: Dict[str, Callable],
    params: Dict[str, Any],
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Run multiple algorithms multiple times for statistical comparison
    
    Args:
        state_space: List of all possible states
        actions: List of all possible actions
        step: Function to take action
        reset: Function to reset environment
        algorithms: List of algorithm names to compare
        algorithm_functions: Dict mapping algorithm names to their functions
        params: Parameters to use for all algorithms
        num_runs: Number of times to run each algorithm
    
    Returns:
        Dictionary containing comparative results and statistics
    """
    results = {}
    
    for algo_name in algorithms:
        if algo_name not in algorithm_functions:
            continue
            
        algo_func = algorithm_functions[algo_name]
        all_rewards = []
        all_lengths = []
        all_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Run the algorithm
            Q, policy, stats = algo_func(
                state_space=state_space,
                actions=actions,
                step=step,
                reset=reset,
                **params
            )
            
            end_time = time.time()
            
            all_rewards.extend(stats['episode_rewards'])
            all_lengths.extend(stats['episode_lengths'])
            all_times.append(end_time - start_time)
        
        # Calculate statistics for this algorithm
        results[algo_name] = {
            'rewards': all_rewards,
            'lengths': all_lengths,
            'times': all_times,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'mean_time': np.mean(all_times),
            'std_time': np.std(all_times),
            'final_reward_mean': np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards),
            'final_reward_std': np.std(all_rewards[-100:]) if len(all_rewards) >= 100 else np.std(all_rewards)
        }
    
    # Perform pairwise statistical tests
    pairwise_tests = perform_pairwise_tests(results, algorithms)
    
    return {
        'results': results,
        'pairwise_tests': pairwise_tests,
        'num_runs': num_runs
    }


def perform_pairwise_tests(results: Dict[str, Any], algorithms: List[str]) -> Dict[str, Any]:
    """
    Perform pairwise t-tests between algorithms
    
    Args:
        results: Results dictionary from run_comparative_analysis
        algorithms: List of algorithm names
    
    Returns:
        Dictionary with pairwise test results
    """
    pairwise_results = {}
    
    for i, algo1 in enumerate(algorithms):
        for algo2 in algorithms[i+1:]:
            if algo1 not in results or algo2 not in results:
                continue
            
            # T-test on final 100 episodes
            rewards1 = results[algo1]['rewards'][-100:] if len(results[algo1]['rewards']) >= 100 else results[algo1]['rewards']
            rewards2 = results[algo2]['rewards'][-100:] if len(results[algo2]['rewards']) >= 100 else results[algo2]['rewards']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
            
            # Determine significance
            alpha = 0.05
            significant = p_value < alpha
            
            pairwise_results[f"{algo1}_vs_{algo2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': significant,
                'better': algo1 if t_stat > 0 else algo2,
                'mean_diff': np.mean(rewards1) - np.mean(rewards2)
            }
    
    return pairwise_results


def calculate_confidence_intervals(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a list of values
    
    Args:
        data: List of values
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0, 0)
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    # Calculate interval
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (mean - h, mean + h)


def generate_comparison_summary(results: Dict[str, Any], algorithms: List[str]) -> str:
    """
    Generate a human-readable summary of comparative results
    
    Args:
        results: Results from run_comparative_analysis
        algorithms: List of algorithm names
    
    Returns:
        Summary string
    """
    summary = []
    summary.append("=" * 80)
    summary.append("COMPARATIVE ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Summary table
    summary.append(f"{'Algorithm':<20} {'Mean Reward':<15} {'Std Reward':<15} {'Mean Time (s)':<15}")
    summary.append("-" * 80)
    
    for algo in algorithms:
        if algo in results:
            r = results[algo]
            summary.append(f"{algo:<20} {r['mean_reward']:<15.4f} {r['std_reward']:<15.4f} {r['mean_time']:<15.4f}")
    
    summary.append("")
    summary.append("STATISTICAL SIGNIFICANCE TESTS (alpha = 0.05)")
    summary.append("-" * 80)
    
    pairwise = results.get('pairwise_tests', {})
    for comparison, test_result in pairwise.items():
        algo1, algo2 = comparison.split('_vs_')
        sig = "**SIGNIFICANT**" if test_result['significant'] else "Not significant"
        better = test_result['better']
        summary.append(f"{algo1} vs {algo2}: p={test_result['p_value']:.4f} ({sig})")
        summary.append(f"  → {better} performs better (diff: {test_result['mean_diff']:.4f})")
    
    summary.append("")
    summary.append("=" * 80)
    
    return "\n".join(summary)
