"""Main content area with tabs"""

import streamlit as st
from interface.tabs import render_training_tab, render_inference_tab, render_design_tab, render_comparison_tab


def render_main_content():
    """Render main content with tabs"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["Training", "Inference", "Design & Analysis", "🆕 Comparative Analysis"])
    
    with tab1:
        render_training_tab()
    
    with tab2:
        render_inference_tab()
    
    with tab3:
        render_design_tab()
    
    with tab4:
        render_comparison_tab()
