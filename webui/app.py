import streamlit as st

st.set_page_config(
    page_title="KARD — Inférence en Direct",
    page_icon="⬜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Reset & Base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    color: #1a1a1a;
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #f8f9fa !important;
    border-right: 1px solid #e0e0e0 !important;
}
[data-testid="stSidebar"] * { color: #333333 !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Brand bar ────────────────────────────── */
.brand-bar {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 0 14px 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 28px;
}
.brand-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px; font-weight: 600;
    color: #1a1a1a; letter-spacing: -0.02em;
}
.brand-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px; color: #666;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.brand-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #1a1a1a; flex-shrink: 0;
}

/* ── Section headings ─────────────────────────── */
.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 500;
    color: #666; letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 8px; margin-bottom: 18px;
}

/* ── Panel blocks ─────────────────────────────── */
.panel {
    background: #f8f9fa; border: 1px solid #e0e0e0;
    border-radius: 4px; padding: 22px 20px;
    margin-bottom: 16px;
}
.panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 500;
    color: #666; margin-bottom: 12px;
}

/* ── Inference live feed ──────────────────────── */
.live-frame {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    min-height: 340px;
    display: flex; align-items: center;
    justify-content: center;
    position: relative; overflow: hidden;
}
.live-badge {
    position: absolute; top: 14px; left: 14px;
    display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.95);
    padding: 4px 10px; border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #666;
    border: 1px solid #e0e0e0;
}
.live-indicator {
    width: 6px; height: 6px; border-radius: 50%;
    background: #1a1a1a;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Confidence bars ──────────────────────────── */
.conf-row {
    display: flex; align-items: center;
    gap: 10px; margin-bottom: 9px;
}
.conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #666;
    width: 180px; flex-shrink: 0;
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
}
.conf-track {
    flex: 1; height: 4px;
    background: #e0e0e0; border-radius: 2px;
    overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 2px;
    background: #1a1a1a;
    transition: width 0.3s ease;
}
.conf-fill.top { background: #000000; }
.conf-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #666;
    width: 38px; text-align: right; flex-shrink: 0;
}

/* ── Predicted label ──────────────────────────── */
.pred-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px; font-weight: 600;
    color: #1a1a1a; letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.pred-conf {
    font-size: 13px; color: #666;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Streamlit overrides ──────────────────────── */
.stMarkdown p, .stMarkdown li {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #666; font-size: 13px;
}
h1,h2,h3,h4 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #1a1a1a !important;
}
.stAlert { background: #f8f9fa !important; border-color: #e0e0e0 !important; }
hr { border-color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load inference page ───────────────────────────────────────────────
from pages import inference; inference.render()
