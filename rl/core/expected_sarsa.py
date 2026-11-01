"""
Expected SARSA Implementation
Improves upon SARSA by using expected value over all actions instead of single action
"""

from typing import List, Callable, Tuple, Dict
import numpy as np
import random


def expected_sarsa(state_space: List[int], actions: List[int], step: Callable, reset: Callable,
                   episodes: int = 1000, alpha: float = 0.1, gamma: float = 1.0,
                   epsilon: float = 0.3) -> Tuple[Dict, Dict, Dict]:
    """
    Expected SARSA algorithm implementation
    
    Args:
        state_space: List of all possible states
        actions: List of all possible actions
        step: Function to take action (state, action) -> (next_state, reward, done)
        reset: Function to reset environment -> initial_state
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate (epsilon-greedy)
    
    Returns:
        Q: Q-function
        policy: Derived policy from Q-function
        stats: Training statistics including rewards and episode lengths
    """
    Q = {(s, a): 0 for s in state_space for a in actions}
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = reset()
        total_reward = 0
        steps = 0
        
        max_steps = 1000
        done = False
        
        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: Q[(state, a)])
            
            next_state, reward, done = step(state, action)
            
            total_reward += reward
            steps += 1
            
            # Calculate expected value for next state
            if done:
                expected_value = 0
            else:
                # Expected value: sum over all actions weighted by their probabilities
                num_actions = len(actions)
                best_action = max(actions, key=lambda a: Q[(next_state, a)])
                
                expected_value = 0
                for a in actions:
                    if a == best_action:
                        # Probability of choosing best action: (1 - epsilon) + epsilon/num_actions
                        prob = (1 - epsilon) + epsilon / num_actions
                    else:
                        # Probability of choosing other actions: epsilon/num_actions
                        prob = epsilon / num_actions
                    expected_value += prob * Q[(next_state, a)]
            
            # Update Q-value using expected value
            target_val = reward + gamma * expected_value
            Q[(state, action)] += alpha * (target_val - Q[(state, action)])
            
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Derive policy from Q-function
    policy = {s: max(actions, key=lambda a: Q[(s, a)]) for s in state_space}
    
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'algorithm': 'Expected SARSA'
    }
    
    return Q, policy, stats
