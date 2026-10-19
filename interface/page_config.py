import streamlit as st


def setup_page():
    """Set up the app shell and shared dark visual system."""
    st.set_page_config(
        page_title="RL Workbench | Interactive Reinforcement Learning Lab",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "An interactive lab for learning, training, and comparing classic reinforcement learning agents.",
        },
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #e8eef8;
            --muted: #91a1b8;
            --surface: #101a2a;
            --surface-raised: #142136;
            --line: rgba(148, 163, 184, .16);
            --mint: #67f3c0;
            --blue: #77b7ff;
        }
        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(ellipse at 8% 0%, rgba(48, 102, 148, .17), transparent 34%),
                radial-gradient(ellipse at 92% 11%, rgba(27, 151, 120, .12), transparent 27%),
                #080e19;
        }
        [data-testid="stHeader"] {
            background: rgba(8, 14, 25, .82);
            backdrop-filter: blur(14px);
        }
        .main .block-container {
            max-width: 1500px;
            padding: 1.5rem 3rem 3rem;
        }
        h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
            color: var(--ink);
        }
        h1, h2, h3 { letter-spacing: -.035em; }
        p, li { color: #b8c4d6; line-height: 1.7; }
        code, pre, .mono { font-family: 'DM Mono', monospace !important; }
        a { color: var(--mint) !important; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e1928 0%, #0a111d 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span { color: #c7d2e2 !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(255,255,255,.025);
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: .45rem;
            border-bottom: 1px solid var(--line);
            background: transparent;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            color: #93a3b8;
            border-radius: 11px 11px 0 0;
            padding: .72rem 1.05rem;
            transition: color .2s ease, background .2s ease;
        }
        [data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: var(--mint);
            background: rgba(103, 243, 192, .07);
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--mint) !important;
            border-bottom: 2px solid var(--mint) !important;
            background: linear-gradient(180deg, rgba(103,243,192,.1), transparent) !important;
        }

        [data-testid="stMetric"] {
            padding: .9rem 1rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(20,33,54,.92), rgba(13,23,38,.92));
        }
        [data-testid="stMetricLabel"] { color: #92a1b5 !important; }
        [data-testid="stMetricValue"] { color: var(--mint) !important; }
        [data-testid="stMetricDelta"] { color: var(--blue) !important; }

        .stButton > button, .stDownloadButton > button {
            color: #07150f !important;
            border: 1px solid rgba(103,243,192,.45);
            border-radius: 11px;
            background: linear-gradient(115deg, #71f0bd, #69cfff);
            font-weight: 800;
            box-shadow: 0 8px 28px rgba(71, 224, 177, .12);
            transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px);
            filter: saturate(1.08);
            box-shadow: 0 12px 34px rgba(71, 224, 177, .23);
        }
        .stButton > button:focus, .stDownloadButton > button:focus {
            box-shadow: 0 0 0 3px rgba(103,243,192,.22);
        }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {
            color: var(--ink);
            background: #111d2e;
            border-color: var(--line);
            border-radius: 10px;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(16, 26, 42, .62);
        }
        [data-testid="stAlert"] { border-radius: 13px; }
        [data-testid="stProgressBar"] > div > div > div > div {
            background: linear-gradient(90deg, var(--mint), var(--blue));
        }
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
            border-radius: 14px;
            overflow: hidden;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.2rem 1.7rem;
            margin: .2rem 0 1.5rem;
            border: 1px solid rgba(119,183,255,.2);
            border-radius: 24px;
            background: linear-gradient(120deg, rgba(17,31,51,.96), rgba(10,20,34,.82));
            box-shadow: 0 20px 70px rgba(0,0,0,.2);
        }
        .hero-shell:after {
            content: '';
            position: absolute;
            width: 270px; height: 270px; right: -58px; top: -154px;
            border-radius: 50%;
            border: 1px solid rgba(103,243,192,.27);
            box-shadow: 0 0 0 24px rgba(103,243,192,.025), 0 0 0 52px rgba(103,243,192,.025), 0 0 80px rgba(103,243,192,.12);
            animation: orbit-glow 8s ease-in-out infinite alternate;
            pointer-events: none;
        }
        .hero-kicker {
            display: inline-flex; align-items: center; gap: .55rem;
            color: var(--mint); font-family: 'DM Mono', monospace;
            font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
        }
        .hero-dot {
            width: 7px; height: 7px; border-radius: 50%; background: var(--mint);
            box-shadow: 0 0 0 5px rgba(103,243,192,.12); animation: beacon 2s ease-in-out infinite;
        }
        .hero-title {
            margin: .75rem 0 .35rem; font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.08; font-weight: 800; color: #eef5ff;
        }
        .hero-title span {
            background: linear-gradient(95deg, #79f1bd, #7ab9ff 80%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-copy { max-width: 800px; margin: 0; color: #aebdd1; font-size: 1rem; }
        .hero-tags { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1.15rem; }
        .hero-tag {
            padding: .35rem .65rem; border: 1px solid var(--line); border-radius: 999px;
            color: #b8c7d9; background: rgba(255,255,255,.035); font-size: .73rem;
        }
        .section-eyebrow {
            margin: .1rem 0 .4rem; color: var(--mint); font: 500 .72rem 'DM Mono', monospace;
            letter-spacing: .1em; text-transform: uppercase;
        }
        .concept-card {
            height: 100%; padding: 1rem 1.1rem; border: 1px solid var(--line);
            border-radius: 15px; background: linear-gradient(145deg, rgba(20,33,54,.8), rgba(13,23,38,.8));
        }
        .concept-card small { color: #8fa1b9; font: .72rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .08em; }
        .concept-card strong { display: block; margin: .4rem 0 .2rem; color: #edf4ff; font-size: 1rem; }
        .concept-card span { color: #aebdd1; font-size: .84rem; line-height: 1.55; }
        .equation-box {
            padding: 1rem 1.2rem; margin: .7rem 0; border: 1px solid rgba(103,243,192,.22);
            border-radius: 13px; background: rgba(103,243,192,.055); color: #c5fbe4;
            font: .91rem 'DM Mono', monospace; overflow-x: auto;
        }
        @keyframes beacon { 0%,100% { box-shadow: 0 0 0 4px rgba(103,243,192,.1); } 50% { box-shadow: 0 0 0 8px rgba(103,243,192,.025); } }
        @keyframes orbit-glow { from { transform: translate(-9px,4px) scale(.97); opacity:.7; } to { transform: translate(8px,13px) scale(1.04); opacity:1; } }
        @media (prefers-reduced-motion: reduce) {
            *, *:before, *:after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; }
        }
        @media (max-width: 850px) {
            .main .block-container { padding: 1.2rem 1.1rem 2rem; }
            .hero-shell { padding: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
