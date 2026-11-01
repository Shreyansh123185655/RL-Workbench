"""
Double Q-Learning Implementation
Reduces overestimation bias in Q-Learning by using two independent Q-functions
"""

from typing import List, Callable, Tuple, Dict
import numpy as np
import random


def double_q_learning(state_space: List[int], actions: List[int], step: Callable, reset: Callable,
                      episodes: int = 1000, alpha: float = 0.1, gamma: float = 1.0,
                      epsilon: float = 0.3) -> Tuple[Dict, Dict, Dict]:
    """
    Double Q-Learning algorithm implementation
    
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
        Q: Combined Q-function (average of Q1 and Q2)
        policy: Derived policy from Q-function
        stats: Training statistics including rewards and episode lengths
    """
    # Initialize two independent Q-functions
    Q1 = {(s, a): 0 for s in state_space for a in actions}
    Q2 = {(s, a): 0 for s in state_space for a in actions}
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = reset()
        total_reward = 0
        steps = 0
        
        max_steps = 1000
        done = False
        
        while not done and steps < max_steps:
            # Epsilon-greedy action selection using combined Q-values
            if np.random.rand() < epsilon:
                action = random.choice(actions)
            else:
                # Use average of Q1 and Q2 for action selection
                combined_Q = {a: (Q1[(state, a)] + Q2[(state, a)]) / 2 for a in actions}
                action = max(actions, key=lambda a: combined_Q[a])
            
            next_state, reward, done = step(state, action)
            
            total_reward += reward
            steps += 1
            
            # Update either Q1 or Q2 randomly (50% chance each)
            if np.random.rand() < 0.5:
                # Update Q1 using Q2 for target
                best_action = max(actions, key=lambda a: Q1[(next_state, a)])
                if done:
                    target_val = reward
                else:
                    target_val = reward + gamma * Q2[(next_state, best_action)]
                Q1[(state, action)] += alpha * (target_val - Q1[(state, action)])
            else:
                # Update Q2 using Q1 for target
                best_action = max(actions, key=lambda a: Q2[(next_state, a)])
                if done:
                    target_val = reward
                else:
                    target_val = reward + gamma * Q1[(next_state, best_action)]
                Q2[(state, action)] += alpha * (target_val - Q2[(state, action)])
            
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Combine Q-functions by averaging
    Q = {(s, a): (Q1[(s, a)] + Q2[(s, a)]) / 2 for s in state_space for a in actions}
    
    # Derive policy from combined Q-function
    policy = {s: max(actions, key=lambda a: Q[(s, a)]) for s in state_space}
    
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'algorithm': 'Double Q-Learning'
    }
    
    return Q, policy, stats
