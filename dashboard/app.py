"""
AIMOS — Real-Time Dashboard
Visualizes all 6 module decisions live using Streamlit.
Run with: streamlit run dashboard/app.py
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import time
import threading
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import deque

from utils.data_collector import AIMOSDataCollector
from modules.control_brain import AIMOSControlBrain

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AIMOS — Adaptive Intelligent Management of OS",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #3B82F6;
        margin: 5px 0;
    }
    .alert-card {
        background: #2d1b1b;
        border-radius: 10px;
        padding: 10px;
        border-left: 4px solid #EF4444;
        margin: 5px 0;
    }
    .ok-card {
        background: #1b2d1b;
        border-radius: 10px;
        padding: 10px;
        border-left: 4px solid #10B981;
        margin: 5px 0;
    }
    .stMetric label { color: #9ca3af !important; }
    .stMetric div[data-testid="stMetricValue"] {
        color: #f9fafb !important;
        font-size: 1.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────
if 'collector' not in st.session_state:
    st.session_state.collector = None
if 'brain' not in st.session_state:
    st.session_state.brain = None
if 'started' not in st.session_state:
    st.session_state.started = False
if 'cpu_history' not in st.session_state:
    st.session_state.cpu_history = deque(maxlen=60)
if 'mem_history' not in st.session_state:
    st.session_state.mem_history = deque(maxlen=60)
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = deque(maxlen=60)
if 'algo_counts' not in st.session_state:
    st.session_state.algo_counts = {
        'FCFS': 0, 'SJF': 0, 'RR': 0, 'PRIORITY': 0
    }
if 'governor_counts' not in st.session_state:
    st.session_state.governor_counts = {
        'performance': 0, 'balanced': 0,
        'powersave': 0, 'adaptive': 0
    }

# ── Helper functions ──────────────────────────────────────────────────────────

def start_aimos():
    """Start collector and control brain."""
    collector = AIMOSDataCollector(interval=0.5)
    collector.start()
    time.sleep(3)  # let data accumulate

    brain = AIMOSControlBrain(collector=collector)
    brain.load_all_models()
    brain.start(interval=1.0)

    st.session_state.collector = collector
    st.session_state.brain     = brain
    st.session_state.started   = True

def stop_aimos():
    if st.session_state.brain:
        st.session_state.brain.stop()
        st.session_state.brain.save_decision_log()
    if st.session_state.collector:
        st.session_state.collector.stop()
        st.session_state.collector.save_to_csv()
    st.session_state.started = False

def get_risk_color(level):
    return {
        'LOW'     : '#10B981',
        'MEDIUM'  : '#F59E0B',
        'HIGH'    : '#EF4444',
        'CRITICAL': '#DC2626',
        'NORMAL'  : '#10B981',
    }.get(level, '#9CA3AF')

def get_algo_color(algo):
    return {
        'FCFS'    : '#3B82F6',
        'SJF'     : '#10B981',
        'RR'      : '#F59E0B',
        'PRIORITY': '#EF4444',
    }.get(algo, '#9CA3AF')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/AIMOS-v1.0-blue", width=120)
    st.title("AIMOS Control")
    st.caption("Adaptive Intelligent Management of OS")
    st.divider()

    if not st.session_state.started:
        if st.button("Start AIMOS", type="primary", use_container_width=True):
            with st.spinner("Starting AIMOS..."):
                start_aimos()
            st.success("AIMOS is running")
            st.rerun()
    else:
        st.success("AIMOS Running")
        if st.button("Stop AIMOS", type="secondary", use_container_width=True):
            stop_aimos()
            st.rerun()

    st.divider()
    st.subheader("System Info")
    import psutil
    st.metric("CPU Cores", psutil.cpu_count())
    st.metric("Total RAM (GB)",
              round(psutil.virtual_memory().total / 1e9, 1))

    st.divider()
    refresh = st.slider("Refresh interval (sec)", 1, 10, 2)

    st.divider()
    st.subheader("Modules")
    modules = [
        ("M1", "CPU Scheduler",  "RL/PPO"),
        ("M2", "Memory Pred.",   "LSTM"),
        ("M3", "Deadlock Pred.", "Random Forest"),
        ("M4", "Disk Optimizer", "K-Means"),
        ("M5", "Anomaly Det.",   "Isolation Forest"),
        ("M6", "Energy Opt.",    "Multi-obj RL"),
    ]
    for mid, name, tech in modules:
        st.markdown(
            f"**{mid}** {name}  \n"
            f"<small style='color:#6B7280'>{tech}</small>",
            unsafe_allow_html=True
        )

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🧠 AIMOS — Adaptive Intelligent Management of OS")
st.caption(
    "Real-time AI-driven OS resource management | "
    "WSL2 Ubuntu | psutil + /proc + eBPF"
)

if not st.session_state.started:
    st.info(
        "Click **Start AIMOS** in the sidebar to begin "
        "real-time AI-driven OS management."
    )

    # Show architecture diagram while waiting
    st.subheader("System Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data Sources**")
        st.code("psutil\n/proc filesystem\nbpftrace eBPF")
    with col2:
        st.markdown("**AI Modules**")
        st.code(
            "M1: CPU Scheduler RL\n"
            "M2: Memory LSTM\n"
            "M3: Deadlock RF\n"
            "M4: Disk K-Means\n"
            "M5: Anomaly IForest\n"
            "M6: Energy RL"
        )
    with col3:
        st.markdown("**OS Actions**")
        st.code(
            "Algorithm switching\n"
            "Memory preloading\n"
            "Deadlock prevention\n"
            "Disk reordering\n"
            "Process isolation\n"
            "CPU freq scaling"
        )
    st.stop()

# ── Live data ─────────────────────────────────────────────────────────────────
brain     = st.session_state.brain
collector = st.session_state.collector
decision  = brain.get_current_decision() if brain else {}
sys_df    = collector.get_system_series(n=60) if collector else pd.DataFrame()
proc_df   = collector.get_latest(n=50) if collector else pd.DataFrame()

# Update history
if decision:
    st.session_state.cpu_history.append(
        decision.get('cpu_percent', 0))
    st.session_state.mem_history.append(
        decision.get('mem_percent', 0))
    st.session_state.anomaly_history.append(
        decision.get('anomaly_count', 0))

    algo = decision.get('cpu_algorithm', 'RR')
    if algo in st.session_state.algo_counts:
        st.session_state.algo_counts[algo] += 1

    gov = decision.get('energy_governor', 'balanced')
    if gov in st.session_state.governor_counts:
        st.session_state.governor_counts[gov] += 1

# ── Row 1: Key metrics ────────────────────────────────────────────────────────
st.subheader("Live System Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)

cpu_val  = decision.get('cpu_percent', 0)
mem_val  = decision.get('mem_percent', 0)
anom_val = decision.get('anomaly_count', 0)
algo_val = decision.get('cpu_algorithm', 'N/A')
gov_val  = decision.get('energy_governor', 'N/A')
cycle    = decision.get('cycle', 0)

c1.metric("CPU Usage",      f"{cpu_val:.1f}%")
c2.metric("Memory Usage",   f"{mem_val:.1f}%")
c3.metric("Anomalies",      str(anom_val),
          delta=f"{'ALERT' if anom_val > 0 else 'OK'}",
          delta_color="inverse")
c4.metric("CPU Algorithm",  algo_val)
c5.metric("Energy Governor",gov_val)
c6.metric("Brain Cycles",   str(cycle))

st.divider()

# ── Row 2: Charts ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("CPU & Memory Over Time")
    cpu_hist  = list(st.session_state.cpu_history)
    mem_hist  = list(st.session_state.mem_history)
    x_vals    = list(range(len(cpu_hist)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=cpu_hist,
        name='CPU %', line=dict(color='#3B82F6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=mem_hist,
        name='Memory %', line=dict(color='#10B981', width=2),
        fill='tozeroy', fillcolor='rgba(16,185,129,0.1)'
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9CA3AF'),
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(showgrid=False, title='Cycles ago'),
        yaxis=dict(showgrid=True, gridcolor='#1f2937',
                   range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Algorithm Selection Distribution")
    algo_data = st.session_state.algo_counts
    colors    = ['#3B82F6','#10B981','#F59E0B','#EF4444']
    fig2 = go.Figure(go.Bar(
        x=list(algo_data.keys()),
        y=list(algo_data.values()),
        marker_color=colors,
        text=list(algo_data.values()),
        textposition='auto',
    ))
    fig2.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9CA3AF'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1f2937'),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 3: Module status panels ───────────────────────────────────────────────
st.subheader("Module Status")
m1, m2, m3, m4, m5, m6 = st.columns(6)

deadlock_color = get_risk_color(decision.get('deadlock_risk', 'LOW'))
mem_color      = get_risk_color(decision.get('memory_pressure', 'LOW'))
anom_color     = '#EF4444' if anom_val > 0 else '#10B981'

with m1:
    st.markdown(f"""
    <div class='metric-card'>
        <small>M1 CPU Scheduler</small><br>
        <b style='font-size:1.1rem'>{algo_val}</b><br>
        <small style='color:#6B7280'>RL/PPO Agent</small>
    </div>""", unsafe_allow_html=True)

with m2:
    mem_pressure = decision.get('memory_pressure', 'LOW')
    st.markdown(f"""
    <div class='metric-card'>
        <small>M2 Memory</small><br>
        <b style='font-size:1.1rem;color:{mem_color}'>{mem_pressure}</b><br>
        <small style='color:#6B7280'>LSTM Predictor</small>
    </div>""", unsafe_allow_html=True)

with m3:
    dl_risk = decision.get('deadlock_risk', 'LOW')
    st.markdown(f"""
    <div class='metric-card'>
        <small>M3 Deadlock</small><br>
        <b style='font-size:1.1rem;color:{deadlock_color}'>{dl_risk}</b><br>
        <small style='color:#6B7280'>Random Forest</small>
    </div>""", unsafe_allow_html=True)

with m4:
    disk_pat = decision.get('disk_pattern', 'N/A')
    st.markdown(f"""
    <div class='metric-card'>
        <small>M4 Disk</small><br>
        <b style='font-size:1.1rem'>{disk_pat}</b><br>
        <small style='color:#6B7280'>K-Means</small>
    </div>""", unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class='metric-card'>
        <small>M5 Security</small><br>
        <b style='font-size:1.1rem;color:{anom_color}'>{anom_val} found</b><br>
        <small style='color:#6B7280'>Isolation Forest</small>
    </div>""", unsafe_allow_html=True)

with m6:
    st.markdown(f"""
    <div class='metric-card'>
        <small>M6 Energy</small><br>
        <b style='font-size:1.1rem'>{gov_val}</b><br>
        <small style='color:#6B7280'>Multi-obj RL</small>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Row 4: Alerts and overrides ───────────────────────────────────────────────
col_alerts, col_overrides = st.columns(2)

with col_alerts:
    st.subheader("Active Alerts")
    alerts = decision.get('alerts', [])
    if alerts:
        for alert in alerts:
            st.markdown(
                f"<div class='alert-card'>⚠ {alert}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div class='ok-card'>✓ No active alerts</div>",
            unsafe_allow_html=True
        )

with col_overrides:
    st.subheader("Control Brain Overrides")
    overrides = decision.get('overrides', [])
    if overrides:
        for ov in overrides:
            st.markdown(
                f"<div class='alert-card'>↺ {ov}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div class='ok-card'>✓ No overrides active</div>",
            unsafe_allow_html=True
        )

st.divider()

# ── Row 5: Live process table ─────────────────────────────────────────────────
st.subheader("Live Process Monitor")
if len(proc_df) > 0:
    display_cols = [
        c for c in ['pid','name','status','cpu_percent',
                    'mem_rss_mb','num_threads','wait_time_ns']
        if c in proc_df.columns
    ]
    display_df = (
        proc_df[display_cols]
        .sort_values('cpu_percent', ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    display_df.columns = [
        c.replace('_', ' ').title()
        for c in display_df.columns
    ]
    st.dataframe(display_df, use_container_width=True, height=300)
else:
    st.info("Waiting for process data...")

# ── Row 6: Anomaly list ───────────────────────────────────────────────────────
anomaly_list = decision.get('anomaly_list', [])
if anomaly_list:
    st.divider()
    st.subheader("⚠ Flagged Processes")
    anom_df = pd.DataFrame(anomaly_list)
    if len(anom_df) > 0:
        st.dataframe(anom_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"AIMOS v1.0 | Last updated: {datetime.now().strftime('%H:%M:%S')} | "
    f"Cycle: {cycle} | "
    f"Data: psutil + /proc + eBPF"
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(refresh)
st.rerun()
