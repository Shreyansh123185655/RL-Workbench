"""
Comparative Analysis Tab
Allows users to compare multiple algorithms with statistical analysis
"""

import streamlit as st
import numpy as np
import time
from rl import create_environment
from rl.core.comparative_analysis import run_comparative_analysis, generate_comparison_summary
from rl.core.qlearning import q_learning
from rl.core.td import td
from rl.core.double_qlearning import double_q_learning
from rl.core.expected_sarsa import expected_sarsa
from interface.utils.comparison_visualizations import (
    plot_algorithm_comparison,
    plot_learning_curves_comparison,
    plot_statistical_significance_heatmap,
    plot_convergence_comparison
)


def render_comparison_tab():
    """Render comparative analysis tab"""
    
    st.markdown("## 🆕 Comparative Analysis")
    st.markdown("Compare multiple algorithms with statistical significance testing")
    
    st.markdown("---")
    
    # Algorithm selection
    st.markdown("### Select Algorithms to Compare")
    
    available_algorithms = ['Q-Learning', 'SARSA', 'Double Q-Learning', 'Expected SARSA']
    selected_algorithms = st.multiselect(
        "Choose 2-4 algorithms",
        available_algorithms,
        default=['Q-Learning', 'SARSA'],
        help="Select algorithms to compare statistically"
    )
    
    if len(selected_algorithms) < 2:
        st.warning("Please select at least 2 algorithms for comparison")
        return
    
    # Parameter configuration
    st.markdown("### Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        episodes = st.number_input("Episodes per run", min_value=100, max_value=10000, value=1000, step=100)
    
    with col2:
        alpha = st.slider("Learning Rate (α)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    
    with col3:
        gamma = st.slider("Discount Factor (γ)", min_value=0.9, max_value=0.99, value=0.99, step=0.01)
    
    with col4:
        epsilon = st.slider("Exploration (ε)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=3, 
                                   help="Number of times to run each algorithm for statistical analysis")
    
    with col2:
        st.info(f"Total episodes: {len(selected_algorithms) * num_runs * episodes}")
    
    # Algorithm function mapping
    algorithm_functions = {
        'Q-Learning': q_learning,
        'SARSA': lambda state_space, actions, step, reset, episodes, alpha, gamma, epsilon: 
                 td(state_space, actions, step, reset, episodes, alpha, gamma, epsilon, 
                    nstep=False, q_based=True),
        'Double Q-Learning': double_q_learning,
        'Expected SARSA': expected_sarsa
    }
    
    # Run comparison
    if st.button("🚀 Run Comparative Analysis", type="primary", use_container_width=True):
        with st.spinner("Running comparative analysis... This may take a few minutes."):
            try:
                # Create environment
                env = create_environment(st.session_state.selected_environment)
                
                # Get environment interface
                state_space = env.state_space
                actions = env.actions
                step_func = env.step
                reset_func = env.reset
                
                # Prepare parameters
                params = {
                    'episodes': episodes,
                    'alpha': alpha,
                    'gamma': gamma,
                    'epsilon': epsilon
                }
                
                # Run comparative analysis
                results = run_comparative_analysis(
                    state_space=state_space,
                    actions=actions,
                    step=step_func,
                    reset=reset_func,
                    algorithms=selected_algorithms,
                    algorithm_functions=algorithm_functions,
                    params=params,
                    num_runs=num_runs
                )
                
                st.session_state.comparison_results = results
                st.session_state.comparison_env = st.session_state.selected_environment
                st.session_state.comparison_params = params
                
                st.success("Comparative analysis completed successfully!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if 'comparison_results' in st.session_state:
        render_comparison_results()


def render_comparison_results():
    """Display comparative analysis results"""
    
    results = st.session_state.comparison_results
    algorithms = list(results['results'].keys())
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Summary text
    summary = generate_comparison_summary(results, algorithms)
    st.text(summary)
    
    # Performance comparison plots
    st.markdown("### Performance Metrics Comparison")
    fig1 = plot_algorithm_comparison(results['results'], algorithms)
    st.pyplot(fig1)
    
    # Learning curves
    st.markdown("### Learning Curves")
    fig2 = plot_learning_curves_comparison(results['results'], algorithms)
    st.pyplot(fig2)
    
    # Statistical significance heatmap
    if len(algorithms) >= 2:
        st.markdown("### Statistical Significance Tests")
        fig3 = plot_statistical_significance_heatmap(results['pairwise_tests'], algorithms)
        st.pyplot(fig3)
    
    # Convergence analysis
    st.markdown("### Convergence Analysis")
    fig4 = plot_convergence_comparison(results['results'], algorithms)
    st.pyplot(fig4)
    
    # Detailed statistics table
    st.markdown("### Detailed Statistics")
    
    stats_data = []
    for algo in algorithms:
        r = results['results'][algo]
        stats_data.append({
            'Algorithm': algo,
            'Mean Reward': f"{r['mean_reward']:.4f}",
            'Std Reward': f"{r['std_reward']:.4f}",
            'Final Reward': f"{r['final_reward_mean']:.4f}",
            'Mean Length': f"{r['mean_length']:.2f}",
            'Mean Time (s)': f"{r['mean_time']:.2f}"
        })
    
    import pandas as pd
    df = pd.DataFrame(stats_data)
    st.table(df)
    
    # Download results
    st.markdown("### Export Results")
    
    import json
    results_json = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="Download Full Results (JSON)",
        data=results_json,
        file_name=f"comparison_results_{st.session_state.comparison_env}.json",
        mime="application/json"
    )
