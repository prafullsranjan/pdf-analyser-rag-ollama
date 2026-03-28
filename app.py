"""
BLS Analyser — People Analytics RAG
Clean, accessible light-theme Streamlit UI.
"""

import streamlit as st
import json
import time
import os
import re
import logging
import warnings
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import io

# ── Silence pdfium shutdown noise (harmless, emitted by docling on Ctrl-C) ─────
warnings.filterwarnings("ignore", message="Cannot close object.*pdfium")
logging.getLogger("pypdfium2").setLevel(logging.CRITICAL)
logging.getLogger("docling").setLevel(logging.WARNING)

# ── Import rag_pipeline first — it creates the timestamped logs/ file handler ──
import rag_pipeline as rag

# ── Reuse the same logger (handlers already attached by rag_pipeline) ───────────
_app_logger = logging.getLogger("rag_pipeline")

_app_logger.info("╔══════════════════════════════════════════════════════════════╗")
_app_logger.info("║        BLS ANALYSER — app.py / Streamlit session started     ║")
_app_logger.info("╚══════════════════════════════════════════════════════════════╝")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLS Analyser · People Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean light theme, high contrast, modern
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════════════
   PALETTE
══════════════════════════════════════════════════════ */
:root {
    --bg:             #f4f6fb;
    --surface:        #ffffff;
    --sidebar-bg:     #1e2235;
    --sidebar-text:   #cdd3e8;
    --sidebar-muted:  #8892b0;
    --border:         #e2e6f0;
    --border-dark:    #d0d6e8;
    --accent:         #3b5bdb;
    --accent-light:   #eef1fd;
    --text:           #111827;
    --text-secondary: #374151;
    --muted:          #6b7280;
    --success:        #059669;
    --success-bg:     #ecfdf5;
    --warning:        #d97706;
    --warning-bg:     #fffbeb;
    --danger:         #dc2626;
    --danger-bg:      #fef2f2;
    --shadow:         0 1px 4px rgba(0,0,0,.07), 0 4px 16px rgba(0,0,0,.05);
    --shadow-md:      0 4px 12px rgba(0,0,0,.1),  0 12px 32px rgba(0,0,0,.07);
}

/* ══════════════════════════════════════════════════════
   GLOBAL BASE — force font, light background, dark text
══════════════════════════════════════════════════════ */
html, body {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px !important;
}

/* Force dark text on every text-bearing element in the MAIN area */
.main p, .main span, .main div, .main label,
.main h1, .main h2, .main h3, .main h4, .main h5,
.main li, .main td, .main th, .main caption,
.element-container p,
.element-container span,
.element-container div,
.element-container label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ══════════════════════════════════════════════════════
   SIDEBAR SHELL
══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0,0,0,.22) !important;
}
/* All text inside sidebar defaults to sidebar-text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: var(--sidebar-text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
/* Sidebar headings — bright white */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
    font-size: .75rem !important;
    font-weight: 700 !important;
    letter-spacing: .13em !important;
    text-transform: uppercase !important;
    margin: 1.2rem 0 .5rem 0 !important;
}
[data-testid="stSidebar"] hr { border-color: #2a2f47 !important; margin: .75rem 0 !important; }

/* ── Sidebar text input ── */
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #252a40 !important;
    border: 1.5px solid #353c5c !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: .87rem !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input::placeholder { color: #4a5278 !important; }
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: #748ffc !important;
    box-shadow: 0 0 0 3px rgba(116,143,252,.2) !important;
}

/* ── Sidebar Multiselect ── */
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {
    background: #252a40 !important;
    border: 1.5px solid #353c5c !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Sidebar input label */
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label span {
    color: var(--sidebar-text) !important;
    font-size: .85rem !important;
}

/* ── Sidebar slider ── */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider label p,
[data-testid="stSidebar"] .stSlider label span,
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
[data-testid="stSidebar"] .stSlider p,
[data-testid="stSidebar"] .stSlider span,
[data-testid="stSidebar"] .stSlider div[data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [aria-valuetext] {
    color: var(--sidebar-text) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #748ffc !important;
    border-color: #748ffc !important;
}

/* ── Sidebar metrics ── */
[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
    color: var(--sidebar-muted) !important;
    font-size: .75rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricDelta"],
[data-testid="stSidebar"] [data-testid="stMetricDelta"] * {
    color: var(--sidebar-muted) !important;
}

/* ── Sidebar expander ── */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    background: #252a40 !important;
    color: var(--sidebar-text) !important;
    border-color: #353c5c !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details > div,
[data-testid="stSidebar"] [data-testid="stExpander"] details > div * {
    background: #1e2235 !important;
    color: var(--sidebar-text) !important;
    border-color: #353c5c !important;
}

/* ── Sidebar alert boxes (st.success / warning / error / info) ── */
[data-testid="stSidebar"] .stAlert,
[data-testid="stSidebar"] .stAlert p,
[data-testid="stSidebar"] .stAlert span,
[data-testid="stSidebar"] .stAlert div,
[data-testid="stSidebar"] [data-testid="stNotification"],
[data-testid="stSidebar"] [data-testid="stNotification"] p,
[data-testid="stSidebar"] [data-testid="stNotification"] span {
    color: #ffffff !important;
}

/* ── Sidebar spinner ── */
[data-testid="stSidebar"] .stSpinner p,
[data-testid="stSidebar"] .stSpinner span,
[data-testid="stSidebar"] .stSpinner div,
[data-testid="stSidebar"] [data-testid="stSpinner"] p,
[data-testid="stSidebar"] [data-testid="stSpinner"] span {
    color: var(--sidebar-text) !important;
}

/* ── Sidebar warning text for empty folder ── */
[data-testid="stSidebar"] [data-testid="stWarningMessage"] p,
[data-testid="stSidebar"] [data-testid="stWarningMessage"] span,
[data-testid="stSidebar"] [data-baseweb="notification"] p,
[data-testid="stSidebar"] [data-baseweb="notification"] span,
[data-testid="stSidebar"] [data-baseweb="notification"] div {
    color: #1a1a1a !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .83rem !important;
    padding: .5rem .8rem !important;
    width: 100% !important;
    transition: all .18s !important;
    color: #ffffff !important;
}

/* ══════════════════════════════════════════════════════
   MAIN AREA — BUTTONS
══════════════════════════════════════════════════════ */
.stButton button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: .92rem !important;
    padding: .6rem 1.75rem !important;
    transition: all .18s !important;
    box-shadow: 0 2px 8px rgba(59,91,219,.28) !important;
}
.stButton button * { color: #ffffff !important; }
.stButton button:hover {
    background: #2f4ac0 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(59,91,219,.38) !important;
}
.stButton button:hover * { color: #ffffff !important; }

/* ══════════════════════════════════════════════════════
   MAIN AREA — TEXT INPUTS & TEXTAREA
══════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-dark) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .93rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,.1) !important;
}
.stTextArea > div > div > textarea::placeholder,
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }
/* Input labels */
.stTextInput label, .stTextInput label p, .stTextInput label span,
.stTextArea  label, .stTextArea  label p, .stTextArea  label span {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}

/* ══════════════════════════════════════════════════════
   MAIN AREA — SLIDER
══════════════════════════════════════════════════════ */
.stSlider label, .stSlider label p, .stSlider label span {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider p, .stSlider span { color: var(--text-secondary) !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════
   MAIN AREA — RADIO BUTTONS
══════════════════════════════════════════════════════ */
.stRadio > div { display: flex !important; flex-wrap: wrap !important; gap: .5rem !important; }
.stRadio label {
    background: var(--surface) !important;
    border: 2px solid var(--border-dark) !important;
    border-radius: 10px !important;
    padding: .45rem 1.1rem !important;
    cursor: pointer !important;
    transition: all .18s !important;
    box-shadow: var(--shadow) !important;
}
.stRadio label,
.stRadio label *,
.stRadio label p,
.stRadio label span,
.stRadio label div,
.stRadio [data-baseweb="radio"] ~ div,
.stRadio [data-baseweb="radio"] ~ div *,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label *,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"],
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] *,
[data-baseweb="radio"] + div,
[data-baseweb="radio"] + div *,
[data-baseweb="radio"] ~ div p,
[data-baseweb="radio"] ~ div span {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stRadio label:hover { border-color: var(--accent) !important; background: var(--accent-light) !important; }
.stRadio label:hover,
.stRadio label:hover * { color: var(--accent) !important; }

/* ══════════════════════════════════════════════════════
   MAIN AREA — SELECTBOX
══════════════════════════════════════════════════════ */
.stSelectbox label, .stSelectbox label p, .stSelectbox label span {
    color: var(--text) !important;
    font-weight: 600 !important;
}
.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] input {
    color: var(--text) !important;
    background: var(--surface) !important;
}
.stSelectbox [data-baseweb="select"] [data-baseweb="base-input"] {
    border-color: var(--border-dark) !important;
    border-radius: 10px !important;
}

/* ══════════════════════════════════════════════════════
   MAIN AREA — EXPANDER
══════════════════════════════════════════════════════ */
.main [data-testid="stExpander"] summary,
.main [data-testid="stExpander"] summary p,
.main [data-testid="stExpander"] summary span,
.main [data-testid="stExpander"] summary svg path,
.main .streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
}
.main [data-testid="stExpander"] summary *,
.main .streamlit-expanderHeader * {
    color: var(--text) !important;
}
.main [data-testid="stExpander"] details > div,
.main .streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}
.main [data-testid="stExpander"] details > div *,
.main .streamlit-expanderContent * {
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.main [data-testid="stExpander"] details[open] summary,
.main [data-testid="stExpander"] details[open] summary p,
.main [data-testid="stExpander"] details[open] summary span,
.main [data-testid="stExpander"] details[open] summary svg path,
.main [data-testid="stExpander"] summary:hover,
.main [data-testid="stExpander"] summary:hover p,
.main [data-testid="stExpander"] summary:hover span,
.main [data-testid="stExpander"] summary:hover svg path {
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ══════════════════════════════════════════════════════
   MAIN AREA — SPINNER
══════════════════════════════════════════════════════ */
[data-testid="stSpinner"] { background: var(--surface) !important; border-radius: 10px !important; }
[data-testid="stSpinner"] *,
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span,
[data-testid="stSpinner"] div,
[data-testid="stSpinner"] > div > div,
.stSpinner p, .stSpinner span, .stSpinner div {
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: .9rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
/* Spinner wheel */
[data-testid="stSpinner"] > div > div > div,
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ══════════════════════════════════════════════════════
   MAIN AREA — ALERTS
══════════════════════════════════════════════════════ */
.stAlert { border-radius: 10px !important; }
/* Force dark text on all alert types in main area */
.main .stAlert,
.main .stAlert *,
.main [data-testid="stNotification"],
.main [data-testid="stNotification"] *,
.main [data-baseweb="notification"],
.main [data-baseweb="notification"] * {
    color: #1a1a1a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.main .stAlert [data-testid="stAlertIcon"] { color: inherit !important; }

/* ══════════════════════════════════════════════════════
   MAIN AREA — METRIC WIDGETS
══════════════════════════════════════════════════════ */
.main [data-testid="stMetricValue"],
.main [data-testid="stMetricValue"] * { color: var(--accent) !important; font-weight: 800 !important; }
.main [data-testid="stMetricLabel"],
.main [data-testid="stMetricLabel"] * { color: var(--text-secondary) !important; font-size: .8rem !important; }
.main [data-testid="stMetricDelta"],
.main [data-testid="stMetricDelta"] * { font-size: .78rem !important; }

/* ══════════════════════════════════════════════════════
   MAIN AREA — TOOLTIP / HELP ICON
══════════════════════════════════════════════════════ */
[data-baseweb="tooltip"] div,
[data-baseweb="tooltip"] span,
[role="tooltip"],
[role="tooltip"] * {
    color: #ffffff !important;
    background: #1e2235 !important;
    font-size: .8rem !important;
    border-radius: 6px !important;
}

/* ══════════════════════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════════════════════ */
.stProgress > div > div { background: var(--accent) !important; }

/* ══════════════════════════════════════════════════════
   MISC STREAMLIT CHROME
══════════════════════════════════════════════════════ */
/* Column containers */
[data-testid="column"] { color: var(--text) !important; }
/* Tab text if any */
[data-baseweb="tab"] p,
[data-baseweb="tab"] span { color: var(--text) !important; }
/* Caption / small text */
.stCaption p, .stCaption span { color: var(--muted) !important; font-size: .8rem !important; }

/* ── Header ── */
.bls-header {
    background: linear-gradient(135deg, #1e2235 0%, #2b3465 55%, #1e2235 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,34,53,.35);
}
.bls-header::after {
    content: '';
    position: absolute; top: -70px; right: -70px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,91,219,.4) 0%, transparent 70%);
    pointer-events: none;
}
.bls-header::before {
    content: '';
    position: absolute; bottom: -90px; left: 32%;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(230,73,128,.22) 0%, transparent 70%);
    pointer-events: none;
}
.bls-badge {
    display: inline-block;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.18);
    color: rgba(255,255,255,.8);
    border-radius: 20px;
    padding: .2rem .8rem;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .8rem;
    position: relative; z-index: 1;
}
.bls-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -.02em;
    margin: 0 0 .25rem 0;
    position: relative; z-index: 1;
}
.bls-logo span {
    background: linear-gradient(90deg, #748ffc, #da77f2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.bls-tagline {
    color: rgba(255,255,255,.6);
    font-size: .9rem;
    margin: 0;
    position: relative; z-index: 1;
    font-weight: 400;
    line-height: 1.5;
}

/* ── Section labels ── */
.sec-label {
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .5rem;
    display: block;
}

/* ── Answer card ── */
.answer-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-left: 5px solid var(--accent);
    border-radius: 14px;
    padding: 1.75rem 2rem;
    line-height: 1.9;
    font-size: .95rem;
    color: var(--text);
    box-shadow: var(--shadow);
    white-space: pre-wrap;
}

/* ── Source pill ── */
.src-pill {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    background: var(--accent-light);
    border: 1.5px solid rgba(59,91,219,.22);
    color: #2f4ac0;
    border-radius: 20px;
    padding: .28rem .8rem;
    font-size: .76rem;
    font-weight: 600;
    margin: .2rem;
}
.src-score {
    background: rgba(59,91,219,.13);
    border-radius: 8px;
    padding: .05rem .38rem;
    font-size: .68rem;
    font-weight: 700;
    color: var(--accent);
}

/* ── Chunk card ── */
.chunk-card {
    background: #f8f9fc;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: .65rem;
}
.chunk-hdr {
    font-size: .77rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: .4rem;
    display: flex; gap: .45rem; align-items: center; flex-wrap: wrap;
}
.chunk-tag {
    background: var(--accent-light);
    border-radius: 6px;
    padding: .1rem .4rem;
    font-size: .7rem;
    font-weight: 600;
    color: #2f4ac0;
}
.chunk-body {
    font-size: .85rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

/* ── KPI card ── */
.kpi-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: transform .15s, box-shadow .15s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.kpi-val  { font-size: 1.9rem; font-weight: 800; color: var(--accent); line-height: 1; margin-bottom: .25rem; }
.kpi-lbl  { font-size: .73rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .09em; margin-bottom: .2rem; }
.kpi-unit { font-size: .71rem; color: var(--muted); }

/* ── Refs section ── */
.refs-box {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: var(--shadow);
}
.refs-ttl {
    font-size: .7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--muted);
    margin-bottom: .7rem;
}

/* ── Sidebar status badges ── */
.sb-ok {
    display: inline-flex; align-items: center; gap: .3rem;
    background: #1a3329; color: #34d399;
    border: 1px solid #16a34a;
    border-radius: 20px; padding: .25rem .75rem;
    font-size: .75rem; font-weight: 700;
}
.sb-warn {
    display: inline-flex; align-items: center; gap: .3rem;
    background: #2d2616; color: #fbbf24;
    border: 1px solid #ca8a04;
    border-radius: 20px; padding: .25rem .75rem;
    font-size: .75rem; font-weight: 700;
}

/* ── History item ── */
.hist-item {
    display: flex; align-items: flex-start; gap: .75rem;
    padding: .65rem .9rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: .4rem;
}
.hist-q    { font-size: .87rem; font-weight: 500; color: var(--text); line-height: 1.4; }
.hist-meta { font-size: .73rem; color: var(--muted); margin-top: .15rem; }

/* ── Sidebar logo ── */
.sb-logo { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.2rem; font-weight:800; color:#fff; margin-bottom:.12rem; }
.sb-logo span { color:#748ffc; }
.sb-sub  { font-size:.66rem; letter-spacing:.15em; text-transform:uppercase; color:#353c5c; font-weight:700; }

/* ── Format hint ── */
.fmt-hint {
    font-size: .8rem; color: var(--text-secondary);
    background: var(--accent-light);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: .45rem .9rem;
    margin-bottom: 1rem;
}

/* ── Divider ── */
.sec-div { border: none; border-top: 1.5px solid var(--border); margin: 1.5rem 0; }

/* ── Footer ── */
.bls-footer {
    text-align: center; color: var(--muted); font-size: .74rem;
    margin-top: 3rem; padding: 1rem 0;
    border-top: 1px solid var(--border);
}

/* ── Code ── */
code {
    background: var(--accent-light) !important;
    color: var(--accent) !important;
    border-radius: 5px !important;
    padding: .1rem .4rem !important;
    font-size: .82rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "folder_path":   "",
        "last_result":   None,
        "last_chunks":   [],
        "index_status":  None,
        "query_history": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.75rem 0 1.25rem 0; border-bottom:1px solid #252a40; margin-bottom:.5rem;">
        <div class="sb-logo">📊 BLS<span> Analyser</span></div>
        <div class="sb-sub">Longitudinal People Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Document Source")

    folder_path = st.text_input(
        "PDF Folder Path",
        value=st.session_state.folder_path,
        placeholder="/path/to/pdf/folder",
        help="Absolute path to folder with PDFs. Sub-folders are scanned recursively.",
    )
    st.session_state.folder_path = folder_path

    col1, col2, col3 = st.columns(3)
    with col1:
        index_btn = st.button("⚡ Index", use_container_width=True, help="Index new or modified PDFs")
    with col2:
        refresh_btn = st.button("🔄 Refresh", use_container_width=True, help="Re-index everything")
    with col3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True, help="Clear all indexed chunks")

    if (index_btn or refresh_btn) and not folder_path:
        st.warning("Enter a folder path first.")
        
    if clear_btn:
        with st.spinner("Clearing the vector index..."):
            res = rag.clear_index()
            st.session_state.index_status = res
            
    if (index_btn or refresh_btn) and folder_path:
        progress_ui = st.empty()
        
        def update_progress(metrics):
            progress_ui.markdown(
                f'<div style="background:#252a40; padding:1rem; border-radius:8px; margin-bottom:1rem; border:1px solid #353c5c;">'
                f'<div style="color:var(--sidebar-muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.6rem;">Processing</div>'
                f'<div style="color:var(--sidebar-text);font-size:.85rem;margin-bottom:.3rem;">📄 PDFs Found: <b style="color:#ffffff;">{metrics["found"]}</b></div>'
                f'<div style="color:var(--sidebar-text);font-size:.85rem;margin-bottom:.3rem;">🟢 PDFs Extracted: <b style="color:#ffffff;">{metrics["extracted"]}</b></div>'
                f'<div style="color:var(--sidebar-text);font-size:.85rem;margin-bottom:.3rem;">✂️ Chunks Created: <b style="color:#ffffff;">{metrics["chunks"]}</b></div>'
                f'<div style="color:var(--sidebar-text);font-size:.85rem;">🧠 Embeddings Saved: <b style="color:#ffffff;">{metrics["embedded"]}</b></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        if index_btn:
            _app_logger.info(f"INDEX triggered: folder={folder_path}")
            with st.spinner("Indexing…"):
                r = rag.index_folder(folder_path, force=False, progress_callback=update_progress)
                st.session_state.index_status = r
            _app_logger.info(f"INDEX complete: {r}")
        if refresh_btn:
            _app_logger.info(f"REFRESH triggered: folder={folder_path}")
            with st.spinner("Re-indexing all files…"):
                r = rag.index_folder(folder_path, force=True, progress_callback=update_progress)
                st.session_state.index_status = r
            _app_logger.info(f"REFRESH complete: {r}")
                
        progress_ui.empty()

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Index Status")
    idx = rag.get_index_status()

    if idx["total_chunks"] == 0:
        st.markdown('<span class="sb-warn">⚠ Not indexed</span>', unsafe_allow_html=True)
        st.markdown(
            '<p>Enter a folder path above and click ⚡ Index to get started.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="sb-ok">● Ready</span>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("PDFs", idx["indexed_files"])
        m2.metric("Chunks", idx["total_chunks"])
        if idx["files"]:
            with st.expander("📄 Indexed Files"):
                for f in idx["files"]:
                    st.markdown(
                        f'<div style="font-size:.8rem;padding:.25rem 0;'
                        f'border-bottom:1px solid #252a40;color:#cdd3e8;">📄 {Path(f).name}</div>',
                        unsafe_allow_html=True,
                    )

    if st.session_state.index_status:
        r = st.session_state.index_status
        if r.get("status") == "success":
            st.success(
                f"**✓ Extraction Complete**\n\n"
                f"**{r.get('indexed', 0)}** PDFs safely extracted & indexed.\n"
                f"**{r.get('skipped', 0)}** skipped (already up to date)."
                + (f"\n**{r.get('removed', 0)}** removed (stale deleted)." if r.get("removed") else "")
            )
        elif r.get("status") == "warning":
            st.warning(r.get("message", "Warning occurred."))
        else:
            st.error(r.get("message", "Error during indexing."))
            
        if r.get("errors"):
            st.error(f"⚠️ {len(r['errors'])} file(s) failed to extract properly.")
            with st.expander("View Error Details", expanded=True):
                for e in r["errors"]:
                    st.markdown(
                        f'<div style="color:#f87171;font-size:.8rem;padding:.3rem 0;border-bottom:1px solid #2a2f47;">'
                        f'• {e}</div>', 
                        unsafe_allow_html=True
                    )

    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    
    # Dynamically extract available years from indexed files
    available_years: set[int] = set()
    for f in idx["files"]:
        match = re.search(r'(19|20)\d{2}', Path(f).name)
        if match:
            available_years.add(int(match.group()))
    available_years_list = sorted(list(available_years))
    
    selected_years = st.multiselect(
        "Filter by Year(s)", 
        options=available_years_list, 
        help="Select specific years to compare (e.g., 1980 vs 2000). Leave blank to search all."
    )
    
    top_k = st.slider("Retrieved Context Chunks", 3, 30, 10,
                      help="Increase this for broad longitudinal comparisons.")
                      
    st.markdown(
        f'<div style="margin-top:.6rem;font-size:.78rem;line-height:2;">'
        f'<span style="color:#4a5278;font-weight:600;">LLM</span>&nbsp;'
        f'<code style="background:#252a40;color:#748ffc;padding:.1rem .4rem;'
        f'border-radius:4px;">{rag.LLM_MODEL}</code><br>'
        f'<span style="color:#4a5278;font-weight:600;">Embed</span>&nbsp;'
        f'<code style="background:#252a40;color:#748ffc;padding:.1rem .4rem;'
        f'border-radius:4px;">{rag.EMBED_MODEL}</code></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bls-header">
    <div class="bls-badge">Historical Labor Analysis · AI Assistant</div>
    <div class="bls-logo">BLS <span>Analyser</span></div>
    <p class="bls-tagline">
        Track job evolution, skill shifts, and workforce trends from 1949–2024.<br>
        Ask questions to compare decades and uncover longitudinal patterns.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Query input
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">Ask a Question</span>', unsafe_allow_html=True)

query = st.text_area(
    "question",
    placeholder=(
        "Compare the accountant role in the 1980 vs. 2000 handbooks.\n"
        "How have education and training requirements for engineers changed since 1950?\n"
        "Tally mentions of 'automation' in job entries."
    ),
    height=115,
    label_visibility="collapsed",
)

st.markdown(
    '<span class="sec-label" style="margin-top:.9rem;display:block;">Answer Format</span>',
    unsafe_allow_html=True,
)
output_format = st.radio(
    "fmt",
    options=["text", "word_cloud", "data_chart", "data_visualization"],
    format_func=lambda x: {
        "text":               "📝  Rich Text & Timeline",
        "word_cloud":         "☁️  Word Cloud",
        "data_chart":         "📊  Data Chart",
        "data_visualization": "📈  KPI Dashboard",
    }[x],
    horizontal=True,
    label_visibility="collapsed",
)

fmt_desc = {
    "text":               "Structured text answer with inline source citations.",
    "word_cloud":         "Answer + word frequency cloud of the most important terms.",
    "data_chart":         "Answer + auto-generated chart from extracted numeric data.",
    "data_visualization": "Answer + KPI metric cards pulled from your documents.",
}
st.markdown(
    f'<div class="fmt-hint">ℹ&nbsp; {fmt_desc[output_format]}</div>',
    unsafe_allow_html=True,
)

generate_btn = st.button("🔍  Generate Insights")

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Answer generation logic
# ─────────────────────────────────────────────────────────────────────────────
if generate_btn:
    if not query.strip():
        st.warning("Please enter a question.")
    elif rag.get_index_status()["total_chunks"] == 0:
        st.error("No documents indexed. Add a folder path in the sidebar and click ⚡ Index.")
    else:
        _app_logger.info(f"QUERY: '{query}' | format={output_format} | years={selected_years} | top_k={top_k}")
        with st.spinner("🔍 Retrieving historical contexts…"):
            chunks = rag.retrieve_chunks(query, n_results=top_k, years=selected_years)
        _app_logger.info(f"RETRIEVAL: {len(chunks)} chunks returned")

        if not chunks:
            _app_logger.warning("No relevant chunks found for the given query and year filters.")
            st.error("No relevant content found for the selected years. Try rephrasing your question or broadening the year filter.")
        else:
            with st.spinner("🧠 Synthesizing longitudinal insights with Gemma 3…"):
                t0     = time.time()
                result = rag.generate_answer(query, output_format=output_format, chunks=chunks)
                elapsed = round(time.time() - t0, 1)
            _app_logger.info(f"GENERATION complete in {elapsed}s | answer_keys={list(result.keys()) if isinstance(result, dict) else type(result)}")

            st.session_state.last_result = result
            st.session_state.last_chunks = chunks
            st.session_state.query_history.insert(0, {
                "query":  query,
                "format": output_format,
                "time":   elapsed,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Render result
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.last_result
chunks = st.session_state.last_chunks

if result:
    answer      = result["answer"]
    sources     = result["sources"]
    fmt         = result.get("output_format", "text")
    format_data = result.get("format_data")

    # ── Answer text ──────────────────────────────────────────────────────────
    st.markdown('<span class="sec-label">Historical Analysis</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)

    # ── Word Cloud ────────────────────────────────────────────────────────────
    if fmt == "word_cloud" and format_data and "word_frequencies" in format_data:
        st.markdown('<span class="sec-label">Word Cloud</span>', unsafe_allow_html=True)
        wf = format_data["word_frequencies"]
        if isinstance(wf, dict) and wf:
            wc = WordCloud(
                width=1000, height=400,
                background_color="#ffffff",
                colormap="Blues",
                max_words=60,
                prefer_horizontal=.8,
                relative_scaling=.6,
                margin=10,
            ).generate_from_frequencies(wf)
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#ffffff")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Word frequency data could not be extracted.")

    # ── Data Chart ────────────────────────────────────────────────────────────
    elif fmt == "data_chart" and format_data and "chart_data" in format_data:
        cd = format_data["chart_data"]
        st.markdown(
            f'<span class="sec-label">{cd.get("title","Data Chart")}</span>',
            unsafe_allow_html=True,
        )
        chart_type = cd.get("chart_type", "bar")
        labels     = cd.get("labels", [])
        values     = cd.get("values", [])
        unit       = cd.get("unit", "")

        if labels and values:
            palette = ["#3b5bdb","#4c6ef5","#748ffc","#91a7ff",
                       "#bac8ff","#e64980","#f06595","#faa2c1"]
            if chart_type == "pie":
                fig = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker_colors=palette,
                    textinfo="label+percent",
                    hole=.4,
                    textfont=dict(size=13, family="Plus Jakarta Sans"),
                ))
            elif chart_type == "line":
                fig = go.Figure(go.Scatter(
                    x=labels, y=values,
                    mode="lines+markers",
                    line=dict(color="#3b5bdb", width=3),
                    marker=dict(size=8, color="#3b5bdb",
                                line=dict(color="#fff", width=2)),
                    fill="tozeroy",
                    fillcolor="rgba(59,91,219,.08)",
                ))
            else:
                fig = go.Figure(go.Bar(
                    x=labels, y=values,
                    marker=dict(color=palette[:len(labels)],
                                line=dict(color="rgba(255,255,255,.5)", width=1)),
                    text=[str(v) for v in values],
                    textposition="outside",
                    textfont=dict(size=11),
                ))

            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(248,249,253,1)",
                font=dict(color="#111827", family="Plus Jakarta Sans", size=12),
                yaxis=dict(gridcolor="#e2e6f0", title=unit,
                           tickfont=dict(size=11), zeroline=False),
                xaxis=dict(gridcolor="#e2e6f0", tickfont=dict(size=11)),
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
                showlegend=(chart_type == "pie"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric chart data could be extracted for this question.")

    # ── KPI Dashboard ─────────────────────────────────────────────────────────
    elif fmt == "data_visualization" and format_data and "viz_data" in format_data:
        vd = format_data["viz_data"]
        st.markdown(
            f'<span class="sec-label">{vd.get("title","KPI Dashboard")}</span>',
            unsafe_allow_html=True,
        )
        metrics = vd.get("metrics", [])
        if metrics:
            n    = min(len(metrics), 8)
            cols = st.columns(min(n, 4))
            t_icon  = {"up": "↑", "down": "↓", "neutral": "→"}
            t_color = {"up": "#059669", "down": "#dc2626", "neutral": "#d97706"}
            for i, m in enumerate(metrics[:n]):
                t = m.get("trend", "neutral")
                cols[i % 4].markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-lbl">{m.get('label','')}</div>
                    <div class="kpi-val">{m.get('value','—')}<span style="font-size:1rem;color:{t_color.get(t,'#d97706')};margin-left:.15rem;">{t_icon.get(t,'→')}</span></div>
                    <div class="kpi-unit">{m.get('unit','')}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No KPI metrics could be extracted for this question.")

    # ── References ────────────────────────────────────────────────────────────
    if sources:
        st.markdown('<span class="sec-label">References</span>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="src-pill">📄 {s["pdf_name"]}'
            f'&nbsp;·&nbsp;p.{s["page"]}'
            f'<span class="src-score">{int(s["score"]*100)}%</span></span>'
            for s in sources
        )
        st.markdown(
            f'<div class="refs-box"><div class="refs-ttl">Source Documents</div>{pills}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View retrieved historical context chunks"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-hdr">
                        <span>Chunk {i}</span>
                        <span class="chunk-tag">📄 {chunk['pdf_name']}</span>
                        <span class="chunk-tag">📅 Year {chunk.get('year', 'N/A')}</span>
                        <span class="chunk-tag">Page {chunk['page']}</span>
                        <span class="chunk-tag">Score: {chunk['score']}</span>
                    </div>
                    <div class="chunk-body">{chunk['text'][:600]}{'…' if len(chunk['text'])>600 else ''}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No references found.")


# ─────────────────────────────────────────────────────────────────────────────
# Query history
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.query_history:
    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    with st.expander("🕑 Recent Queries"):
        icons = {"text":"📝","word_cloud":"☁️","data_chart":"📊","data_visualization":"📈"}
        for h in st.session_state.query_history[:10]:
            icon = icons.get(h["format"], "📝")
            st.markdown(f"""
            <div class="hist-item">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <div class="hist-q">{h['query'][:100]}{'…' if len(h['query'])>100 else ''}</div>
                    <div class="hist-meta">{h['format'].replace('_',' ').title()} &nbsp;·&nbsp; {h['time']}s</div>
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bls-footer">
    BLS Analyser &nbsp;·&nbsp; Historical People Analytics &nbsp;·&nbsp;
    Powered by Gemma 3 &nbsp;+&nbsp; Ollama &nbsp;+&nbsp; ChromaDB
</div>
""", unsafe_allow_html=True)