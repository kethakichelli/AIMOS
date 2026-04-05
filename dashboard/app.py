"""
AIMOS — Live Dashboard
Run with: streamlit run dashboard/app.py
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
    layout="wide"
)

st.title("🧠 AIMOS — AI-Driven OS Management System")
st.caption("Real-time AI monitoring of CPU, Memory, Disk, Security & Energy")

# ── Live system metrics ───────────────────────────────────────────────────────
st.subheader("Live System Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

cpu    = psutil.cpu_percent(interval=0.5)
mem    = psutil.virtual_memory()
disk   = psutil.disk_usage('/')
net    = psutil.net_io_counters()
load   = psutil.getloadavg()[0]

col1.metric("CPU Usage",      f"{cpu:.1f}%",
            delta=f"load {load:.2f}")
col2.metric("Memory Used",    f"{mem.percent:.1f}%",
            delta=f"{mem.used//1e6:.0f} MB")
col3.metric("Disk Used",      f"{disk.percent:.1f}%",
            delta=f"{disk.used//1e9:.1f} GB")
col4.metric("Net Sent",       f"{net.bytes_sent//1e6:.0f} MB")
col5.metric("Net Recv",       f"{net.bytes_recv//1e6:.0f} MB")

st.divider()

# ── Module results ────────────────────────────────────────────────────────────
st.subheader("AI Module Results")

tab1, tab2, tab3, tab4 = st.tabs([
    "CPU Scheduler", "Memory Predictor",
    "Deadlock Detector", "Disk Optimizer"
])

# CPU Scheduler tab
with tab1:
    st.markdown("### Module 1 — PPO Reinforcement Learning Scheduler")
    st.markdown("""
    **How it works:** A PPO agent observes 7 real system features
    (CPU load, queue length, burst ratio, I/O ratio...) and selects
    the best scheduling algorithm dynamically.

    **Innovation:** The OS no longer uses a fixed algorithm —
    the AI switches between FCFS, SJF, Round Robin, and Priority
    based on current workload.
    """)

    action_csv = os.path.join(RESULT_DIR, 'cpu_scheduler_actions.csv')
    if os.path.exists(action_csv):
        df = pd.read_csv(action_csv)
        col1, col2 = st.columns(2)
        with col1:
            counts = df['algorithm'].value_counts().reset_index()
            counts.columns = ['Algorithm', 'Count']
            fig = px.pie(counts, names='Algorithm', values='Count',
                        title='Algorithm Selection Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(df.head(200), y='reward',
                          title='Agent Reward Over Time',
                          labels={'reward':'Reward','index':'Step'},
                          color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig2, use_container_width=True)

        base_csv = os.path.join(RESULT_DIR, 'cpu_baseline_comparison.csv')
        if os.path.exists(base_csv):
            st.markdown("**Baseline Comparison**")
            bdf = pd.read_csv(base_csv)
            st.dataframe(bdf, use_container_width=True)
    else:
        st.info("Run train_cpu_scheduler.py to generate results")

# Memory tab
with tab2:
    st.markdown("### Module 2 — LSTM Page Fault Predictor")
    st.markdown("""
    **How it works:** An LSTM network observes the last 10 page accesses
    and predicts which pages will be needed next — preloading them
    before a page fault occurs.

    **Innovation:** Proactive memory management vs reactive replacement.
    """)

    mem_json = os.path.join(RESULT_DIR, 'memory_results.json')
    if os.path.exists(mem_json):
        with open(mem_json) as f:
            mr = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Page Fault Reduction vs LRU",
                    f"{mr['reduction_vs_lru_pct']}%")
        col2.metric("Top-1 Prediction Accuracy",
                    f"{mr['top1_accuracy']}%")
        col3.metric("Top-3 Prediction Accuracy",
                    f"{mr['top3_accuracy']}%")

        fig = go.Figure(data=[
            go.Bar(name='FIFO',      x=['Page Faults'],
                   y=[mr['fifo_faults']], marker_color='#e74c3c'),
            go.Bar(name='LRU',       x=['Page Faults'],
                   y=[mr['lru_faults']],  marker_color='#e67e22'),
            go.Bar(name='AI (AIMOS)',x=['Page Faults'],
                   y=[mr['ai_faults']],   marker_color='#27ae60'),
        ])
        fig.update_layout(title='Page Faults: AI vs Traditional',
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run memory_lstm.py to generate results")

# Deadlock tab
with tab3:
    st.markdown("### Module 3 — Random Forest Deadlock Predictor")
    st.markdown("""
    **How it works:** A Random Forest classifier trained on 8,000
    resource allocation states (labeled safe/unsafe by Banker's Algorithm)
    predicts deadlock risk as a probability score 0.0–1.0.

    **Innovation:** Outputs continuous risk score, not just binary safe/unsafe.
    Detects danger before allocation is granted.
    """)

    dl_json = os.path.join(RESULT_DIR, 'deadlock_results.json')
    if os.path.exists(dl_json):
        with open(dl_json) as f:
            dr = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Random Forest ROC-AUC", str(dr['rf_roc_auc']))
        col2.metric("SVM ROC-AUC",           str(dr['svm_roc_auc']))
        col3.metric("Detection Approach",    "Probabilistic Risk Score")

        # Risk gauge demo
        demo_risk = 0.73
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=demo_risk * 100,
            title={'text': "Deadlock Risk Score (demo)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c"},
                'steps': [
                    {'range': [0,  50], 'color': '#2ecc71'},
                    {'range': [50, 75], 'color': '#f39c12'},
                    {'range': [75,100], 'color': '#e74c3c'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 'value': 75
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run deadlock_predictor.py to generate results")

# Disk tab
with tab4:
    st.markdown("### Module 4 — K-Means Disk Optimizer")
    st.markdown("""
    **How it works:** K-Means clustering groups files by co-access frequency.
    When any file in a cluster is accessed, the AI prefetches the whole cluster.
    Disk requests are reordered by cluster proximity — elevator style.

    **Innovation:** Combines access pattern learning with seek time optimization.
    """)

    disk_json = os.path.join(RESULT_DIR, 'disk_results.json')
    if os.path.exists(disk_json):
        with open(disk_json) as f:
            dkr = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("SCAN Seek Time",  f"{dkr['scan_seek']} cyl")
        col2.metric("SSTF Seek Time",  f"{dkr['sstf_seek']} cyl")
        col3.metric("AI Seek Time",    f"{dkr['ai_seek']} cyl",
                    delta=f"{dkr['reduction_vs_best_pct']}% vs best")

        fig = px.bar(
            x=['SCAN', 'SSTF', 'AI (AIMOS)'],
            y=[dkr['scan_seek'], dkr['sstf_seek'], dkr['ai_seek']],
            color=['SCAN', 'SSTF', 'AI (AIMOS)'],
            color_discrete_map={
                'SCAN':'#e74c3c','SSTF':'#e67e22','AI (AIMOS)':'#27ae60'},
            title='Total Seek Time Comparison',
            labels={'x':'Algorithm','y':'Cylinders Traversed'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run disk_optimizer.py to generate results")

st.divider()

# ── Live process table ────────────────────────────────────────────────────────
st.subheader("Live Process Monitor")
procs = []
for p in psutil.process_iter(
        ['pid','name','cpu_percent','memory_percent','status']):
    try:
        procs.append(p.info)
    except:
        pass

proc_df = pd.DataFrame(procs).sort_values(
    'cpu_percent', ascending=False).head(15)
st.dataframe(proc_df, use_container_width=True)

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} — "
           f"AIMOS v1.0 | 18 cores | WSL2 Ubuntu")

if st.button("Refresh"):
    st.rerun()
