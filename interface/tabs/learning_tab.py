"""Interactive concepts and method notes for the RL Workbench."""

import streamlit as st

from rl.settings import ALGORITHMS, ENVIRONMENTS


FORMULAS = {
    "Value Iteration": "Vₖ₊₁(s) = maxₐ Σₛ′ P(s′|s,a) [ r + γVₖ(s′) ]",
    "Policy Iteration": "Evaluate π: Vπ(s)  →  Improve π: argmaxₐ Σₛ′ P(s′|s,a)[r + γVπ(s′)]",
    "Monte Carlo": "Gₜ = rₜ₊₁ + γrₜ₊₂ + γ²rₜ₊₃ + …  ·  update after the episode ends",
    "TD(0)": "V(s) ← V(s) + α [ r + γV(s′) − V(s) ]",
    "n-step TD": "Gₜ:ₜ₊ₙ = rₜ₊₁ + … + γⁿV(sₜ₊ₙ)  ·  blend rewards with a bootstrap",
    "SARSA": "Q(s,a) ← Q(s,a) + α [ r + γQ(s′,a′) − Q(s,a) ]",
    "Q-Learning": "Q(s,a) ← Q(s,a) + α [ r + γ maxₐ′Q(s′,a′) − Q(s,a) ]",
}


def render_learning_tab():
    """Show a method-aware explanation and a live Bellman update sandbox."""
    st.markdown('<div class="section-eyebrow">01 / Learn the loop</div>', unsafe_allow_html=True)
    st.header("A policy is learned one experience at a time")
    st.write(
        "An agent observes a state, chooses an action, receives feedback, and uses that experience to improve future decisions. "
        "Select a method below to see what gets updated and try a single value update yourself."
    )

    flow = st.columns(4)
    steps = [
        ("01 · OBSERVE", "State", "Where am I, and what can I see?"),
        ("02 · DECIDE", "Action", "Which move should I try?"),
        ("03 · FEEDBACK", "Reward", "Was the outcome useful?"),
        ("04 · IMPROVE", "Update", "How should I act next time?"),
    ]
    for col, (eyebrow, title, desc) in zip(flow, steps):
        with col:
            st.markdown(
                f'<div class="concept-card"><small>{eyebrow}</small><strong>{title}</strong><span>{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Choose a learning method")
    selected = st.selectbox(
        "Algorithm",
        list(ALGORITHMS),
        index=list(ALGORITHMS).index(st.session_state.selected_algorithm),
        key="learn_algorithm_selector",
    )
    info = ALGORITHMS[selected]
    env_info = ENVIRONMENTS[st.session_state.selected_environment]

    info_col, context_col = st.columns([1.3, 1])
    with info_col:
        st.markdown(f"#### {selected}")
        st.write(info["description"])
        st.markdown(
            f'<div class="equation-box">{FORMULAS[selected]}</div>',
            unsafe_allow_html=True,
        )
        st.caption("The formula is the update rule at the center of this method. α controls how much a new estimate changes the old one; γ controls the weight of future reward.")
    with context_col:
        st.markdown("#### Your current experiment")
        st.markdown(
            f"**Environment:** {st.session_state.selected_environment}  \n"
            f"**Problem:** {env_info['description']}  \n"
            f"**State space:** {env_info['state_space']}  \n"
            f"**Action space:** {env_info['action_space']}  \n"
            f"**Method family:** {info['type']}"
        )
        st.info("Use the left controls to choose an environment and tune the experiment. Then open **Train** to collect rewards and inspect the learned policy.")

    st.markdown("### Try one Q-value update")
    st.write("Change the values and watch the estimate move toward its target. This sandbox mirrors the Q-Learning update; SARSA uses the value of the next action actually taken instead of the maximum next-action value.")
    control_cols = st.columns(4)
    with control_cols[0]:
        old_q = st.slider("Current Q(s, a)", -5.0, 10.0, 1.0, 0.5, key="sandbox_old_q")
    with control_cols[1]:
        reward = st.slider("Observed reward r", -5.0, 10.0, 2.0, 0.5, key="sandbox_reward")
    with control_cols[2]:
        next_q = st.slider("Best next Q(s′, a′)", -5.0, 10.0, 4.0, 0.5, key="sandbox_next_q")
    with control_cols[3]:
        alpha = st.slider("Learning rate α", 0.0, 1.0, 0.25, 0.05, key="sandbox_alpha")

    gamma = st.slider("Discount factor γ", 0.0, 1.0, 0.90, 0.05, key="sandbox_gamma")
    target = reward + gamma * next_q
    updated = old_q + alpha * (target - old_q)
    result_cols = st.columns(3)
    result_cols[0].metric("One-step target", f"{target:.2f}")
    result_cols[1].metric("Updated Q(s, a)", f"{updated:.2f}", delta=f"{updated - old_q:+.2f}")
    result_cols[2].metric("Change from old estimate", f"{updated - old_q:+.2f}")
    st.markdown(
        f'<div class="equation-box">Q_new = {old_q:.2f} + {alpha:.2f} × [({reward:.2f} + {gamma:.2f} × {next_q:.2f}) − {old_q:.2f}] = {updated:.2f}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Connect the demo to your course report"):
        st.write("Use this workbench to support the report sections in the course brief. Document the problem and objective, state/action/reward design, algorithm and hyperparameters, training strategy, comparative results, limitations, and future work. Cite any external code, data, or figures, and verify every result you report.")
        checklist = st.columns(3)
        checklist[0].checkbox("Problem and objective", key="report_problem")
        checklist[1].checkbox("Method and experiment setup", key="report_method")
        checklist[2].checkbox("Results, limits, references", key="report_results")
