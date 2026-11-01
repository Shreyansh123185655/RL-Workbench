"""Interface Tabs - Training, Inference, Design/History, and Comparison tabs"""

from interface.tabs.training_tab import render_training_tab
from interface.tabs.inference_tab import render_inference_tab
from interface.tabs.design_tab import render_design_tab
from interface.tabs.comparison_tab import render_comparison_tab

__all__ = [
    'render_training_tab',
    'render_inference_tab',
    'render_design_tab',
    'render_comparison_tab'
]
