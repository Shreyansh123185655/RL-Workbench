from rl.core.wrappers.value_iteration import ValueIterationAgent
from rl.core.wrappers.policy_iteration import PolicyIterationAgent
from rl.core.wrappers.monte_carlo import MonteCarloAgent
from rl.core.wrappers.qlearning import QLearningAgent
from rl.core.wrappers.td0 import TD0Agent
from rl.core.wrappers.sarsa import SARSAAgent
from rl.core.wrappers.nstep_td import NStepTDAgent
from rl.core.wrappers.double_qlearning import DoubleQLearningAgent
from rl.core.wrappers.expected_sarsa import ExpectedSARSAAgent

def create_agent(algo, env, **params):
    
    AGENTS = {
        'Value Iteration': ValueIterationAgent,
        'Policy Iteration': PolicyIterationAgent,
        'Monte Carlo': MonteCarloAgent,
        'Q-Learning': QLearningAgent,
        'TD(0)': TD0Agent,
        'SARSA': SARSAAgent,
        'n-step TD': NStepTDAgent,
        'Double Q-Learning': DoubleQLearningAgent,
        'Expected SARSA': ExpectedSARSAAgent,
    }
    
    if algo not in AGENTS:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return AGENTS[algo](env, **params)
