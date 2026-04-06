"""
AIMOS — Live Dashboard with auto-refresh
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import psutil, os, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/AIMOS"))
from utils.config import RESULT_DIR, DATA_DIR

st.set_page_config(
    page_title="AIMOS Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Auto-refresh every 2 seconds ──────────────────────────────────────────────
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0
st.session_state.refresh_count += 1

st.title("🧠 AIMOS — AI-Driven OS Management System")
st.caption(
    f"Live update #{st.session_state.refresh_count} — "
    f"{datetime.now().strftime('%H:%M:%S')} — "
    f"Auto-refreshing every 2 seconds"
)

# ── Live system metrics ───────────────────────────────────────────────────────
cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
cpu_avg      = sum(cpu_per_core) / len(cpu_per_core)
mem          = psutil.virtual_memory()
disk         = psutil.disk_usage('/')
net          = psutil.net_io_counters()
load         = psutil.getloadavg()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("CPU (avg)",     f"{cpu_avg:.1f}%",
            delta=f"load {load[0]:.2f}")
col2.metric("Memory",        f"{mem.percent:.1f}%",
            delta=f"{mem.used//1_000_000:.0f} MB used")
col3.metric("Disk",          f"{disk.percent:.1f}%",
            delta=f"{disk.used//1_000_000_000:.1f} GB used")
col4.metric("Net sent",      f"{net.bytes_sent//1_000_000:.0f} MB")
col5.metric("Net recv",      f"{net.bytes_recv//1_000_000:.0f} MB")

st.divider()

# ── CPU core chart ────────────────────────────────────────────────────────────
st.subheader("CPU — Per Core Usage")
fig_cores = go.Figure(go.Bar(
    x=[f"Core {i}" for i in range(len(cpu_per_core))],
    y=cpu_per_core,
    marker_color=[
        '#e74c3c' if v > 80 else '#f39c12' if v > 50 else '#27ae60'
        for v in cpu_per_core
    ]
))
fig_cores.update_layout(
    height=200, margin=dict(t=10, b=10, l=10, r=10),
    yaxis=dict(range=[0, 100], title="Usage %"),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_cores, use_container_width=True)

st.divider()

# ── AI Module Results ─────────────────────────────────────────────────────────
st.subheader("AI Module Decisions")

tab1, tab2, tab3, tab4 = st.tabs([
    "CPU Scheduler", "Memory Predictor",
    "Deadlock Detector", "Disk Optimizer"
])

with tab1:
    st.markdown("**Module 1 — PPO Reinforcement Learning Scheduler**")
    action_csv = os.path.join(RESULT_DIR, 'cpu_scheduler_actions.csv')
    if os.path.exists(action_csv):
        df = pd.read_csv(action_csv)
        col1, col2 = st.columns(2)
        with col1:
            if 'algorithm' in df.columns:
                counts = df['algorithm'].value_counts().reset_index()
                counts.columns = ['Algorithm', 'Count']
                fig = px.pie(
                    counts, names='Algorithm', values='Count',
                    title='Algorithm Selection Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                col_to_plot = numeric_cols[0]
                fig2 = px.line(
                    df.head(200), y=col_to_plot,
                    title=f'Agent metric: {col_to_plot}',
                    color_discrete_sequence=['#2ecc71']
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

        base_csv = os.path.join(RESULT_DIR, 'cpu_baseline_comparison.csv')
        if os.path.exists(base_csv):
            st.markdown("**Baseline comparison**")
            st.dataframe(pd.read_csv(base_csv), use_container_width=True)
    else:
        st.info("Run train_cpu_scheduler.py to generate results")

with tab2:
    st.markdown("**Module 2 — LSTM Page Fault Predictor**")
    mem_json = os.path.join(RESULT_DIR, 'memory_results.json')
    if os.path.exists(mem_json):
        with open(mem_json) as f:
            mr = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Page fault reduction vs LRU",
                  f"{mr['reduction_vs_lru_pct']}%")
        c2.metric("Top-1 accuracy", f"{mr['top1_accuracy']}%")
        c3.metric("Top-3 accuracy", f"{mr['top3_accuracy']}%")

        fig = go.Figure(data=[
            go.Bar(name='FIFO', x=['Faults'],
                   y=[mr['fifo_faults']], marker_color='#e74c3c'),
            go.Bar(name='LRU',  x=['Faults'],
                   y=[mr['lru_faults']],  marker_color='#e67e22'),
            go.Bar(name='AI',   x=['Faults'],
                   y=[mr['ai_faults']],   marker_color='#27ae60'),
        ])
        fig.update_layout(
            title='Page faults: AI vs traditional',
            barmode='group', height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run memory_lstm.py to generate results")

with tab3:
    st.markdown("**Module 3 — Random Forest Deadlock Predictor**")
    dl_json = os.path.join(RESULT_DIR, 'deadlock_results.json')
    if os.path.exists(dl_json):
        with open(dl_json) as f:
            dr = json.load(f)
        c1, c2 = st.columns(2)
        c1.metric("RF ROC-AUC",  str(dr['rf_roc_auc']))
        c2.metric("SVM ROC-AUC", str(dr['svm_roc_auc']))

        demo_risk = 0.73
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(demo_risk * 100, 1),
            title={'text': "Deadlock risk score (demo)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "#e74c3c"},
                'steps': [
                    {'range': [0,  50], 'color': '#2ecc71'},
                    {'range': [50, 75], 'color': '#f39c12'},
                    {'range': [75,100], 'color': '#e74c3c'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75, 'value': 75
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run deadlock_predictor.py to generate results")

with tab4:
    st.markdown("**Module 4 — K-Means Disk Optimizer**")
    disk_json = os.path.join(RESULT_DIR, 'disk_results.json')
    if os.path.exists(disk_json):
        with open(disk_json) as f:
            dkr = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("SCAN seek time",  f"{dkr['scan_seek']} cyl")
        c2.metric("SSTF seek time",  f"{dkr['sstf_seek']} cyl")
        c3.metric("AI seek time",    f"{dkr['ai_seek']} cyl",
                  delta=f"{dkr['reduction_vs_best_pct']}% vs best")

        fig = px.bar(
            x=['SCAN', 'SSTF', 'AI (AIMOS)'],
            y=[dkr['scan_seek'], dkr['sstf_seek'], dkr['ai_seek']],
            color=['SCAN', 'SSTF', 'AI (AIMOS)'],
            color_discrete_map={
                'SCAN':'#e74c3c',
                'SSTF':'#e67e22',
                'AI (AIMOS)':'#27ae60'
            },
            title='Seek time comparison',
            labels={'x': 'Algorithm', 'y': 'Cylinders traversed'}
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run disk_optimizer.py to generate results")

st.divider()

# ── Live process table ────────────────────────────────────────────────────────
st.subheader("Live Process Monitor")
procs = []
for p in psutil.process_iter(
        ['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
    try:
        info = p.info
        if info['cpu_percent'] is not None:
            procs.append(info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

proc_df = (pd.DataFrame(procs)
           .sort_values('cpu_percent', ascending=False)
           .head(15)
           .reset_index(drop=True))

proc_df['cpu_percent']    = proc_df['cpu_percent'].round(2)
proc_df['memory_percent'] = proc_df['memory_percent'].round(2)
st.dataframe(proc_df, use_container_width=True)

# ── Raw metrics from collector ────────────────────────────────────────────────
raw_csv = os.path.join(DATA_DIR, 'raw_metrics.csv')
if os.path.exists(raw_csv):
    st.subheader("Collected OS Metrics (last 20 snapshots)")
    try:
        raw = pd.read_csv(raw_csv).tail(20)
        if 'cpu_percent' in raw.columns and 'timestamp' in raw.columns:
            fig_cpu = px.line(
                raw, x='timestamp', y='cpu_percent',
                title='CPU % over time',
                color_discrete_sequence=['#185FA5']
            )
            fig_cpu.update_layout(
                height=200,
                margin=dict(t=30, b=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
    except Exception:
        pass

st.divider()
st.caption(
    f"AIMOS v2.0 | 18 cores | WSL2 Ubuntu | "
    f"github.com/kethakichelli/AIMOS"
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(2)
st.rerun()
