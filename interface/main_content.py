"""Top-level navigation for the learning and experimentation workspace."""

import streamlit as st

from interface.tabs import (
    render_design_tab,
    render_inference_tab,
    render_learning_tab,
    render_training_tab,
)


def render_main_content():
    """Render a guided path from concepts through analysis."""
    learn, train, inference, analysis = st.tabs(
        ["01 · Learn", "02 · Train", "03 · Evaluate", "04 · Analyze"]
    )

    with learn:
        render_learning_tab()
    with train:
        render_training_tab()
    with inference:
        render_inference_tab()
    with analysis:
        render_design_tab()
