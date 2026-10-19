"""Core package for the RL Workbench algorithms and environments."""

from rl.core.wrappers import create_agent
from rl.envs import create_environment

__version__ = '1.0.0'

__all__ = ['create_agent', 'create_environment']
