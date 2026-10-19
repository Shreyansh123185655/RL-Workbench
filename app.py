import streamlit as st

from interface.main_content import render_main_content
from interface.page_config import setup_page
from interface.sidebar import render_sidebar
from rl.settings import ALGORITHMS, ENVIRONMENTS


def render_hero():
    """Present the project and its learning goal at a glance."""
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-kicker"><span class="hero-dot"></span>Reinforcement learning · Interactive lab</div>
            <h1 class="hero-title">From rewards to <span>reliable policies.</span></h1>
            <p class="hero-copy">Build intuition by changing an environment, training an agent, and tracing how experience becomes a policy. Explore classic tabular methods with visible states, actions, rewards, and learning curves.</p>
            <div class="hero-tags">
                <span class="hero-tag">{len(ALGORITHMS)} classic algorithms</span>
                <span class="hero-tag">{len(ENVIRONMENTS)} environments</span>
                <span class="hero-tag">Train · inspect · compare</span>
                <span class="hero-tag">Course project edition</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Run the reinforcement learning learning and experimentation workbench."""
    setup_page()

    defaults = {
        "selected_environment": "GridWorld",
        "selected_algorithm": "Q-Learning",
        "training_complete": False,
        "trained_agent": None,
        "training_history": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    render_sidebar()
    render_hero()
    render_main_content()

    st.markdown(
        """
        <div style="text-align:center; margin-top:2.5rem; padding:1rem 0; border-top:1px solid rgba(148,163,184,.12); color:#7f90a8; font-size:.78rem">
            <span style="color:#67f3c0">RL WORKBENCH</span> &nbsp;·&nbsp; Learn by changing one decision at a time
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
