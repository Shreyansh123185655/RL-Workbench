"""
Test script for new features
"""

import sys
sys.path.append('/Users/shreyanshgupta/RL-Workbench')

# Test imports
print("Testing imports...")
try:
    from rl.core.double_qlearning import double_q_learning
    print("✓ double_qlearning imported successfully")
except Exception as e:
    print(f"✗ Failed to import double_qlearning: {e}")

try:
    from rl.core.expected_sarsa import expected_sarsa
    print("✓ expected_sarsa imported successfully")
except Exception as e:
    print(f"✗ Failed to import expected_sarsa: {e}")

try:
    from rl.core.comparative_analysis import run_comparative_analysis
    print("✓ comparative_analysis imported successfully")
except Exception as e:
    print(f"✗ Failed to import comparative_analysis: {e}")

try:
    from interface.utils.comparison_visualizations import plot_algorithm_comparison
    print("✓ comparison_visualizations imported successfully")
except Exception as e:
    print(f"✗ Failed to import comparison_visualizations: {e}")

try:
    from interface.tabs.comparison_tab import render_comparison_tab
    print("✓ comparison_tab imported successfully")
except Exception as e:
    print(f"✗ Failed to import comparison_tab: {e}")

try:
    from rl.core.wrappers.double_qlearning import DoubleQLearningAgent
    print("✓ DoubleQLearningAgent imported successfully")
except Exception as e:
    print(f"✗ Failed to import DoubleQLearningAgent: {e}")

try:
    from rl.core.wrappers.expected_sarsa import ExpectedSARSAAgent
    print("✓ ExpectedSARSAAgent imported successfully")
except Exception as e:
    print(f"✗ Failed to import ExpectedSARSAAgent: {e}")

print("\nAll imports successful!")
