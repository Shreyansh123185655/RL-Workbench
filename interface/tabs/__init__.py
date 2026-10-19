"""Interface tabs for learning, training, inference, and analysis."""

from interface.tabs.learning_tab import render_learning_tab
from interface.tabs.training_tab import render_training_tab
from interface.tabs.inference_tab import render_inference_tab
from interface.tabs.design_tab import render_design_tab

__all__ = [
    'render_learning_tab',
    'render_training_tab',
    'render_inference_tab',
    'render_design_tab'
]
