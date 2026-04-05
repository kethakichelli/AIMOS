"""
AIMOS — Adaptive Intelligent Management of OS
Central configuration file.
All modules import from here.
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.expanduser("~/AIMOS")

# ── Directory paths ───────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODEL_DIR   = os.path.join(ROOT_DIR, "models")
MODULE_DIR  = os.path.join(ROOT_DIR, "modules")
LOG_DIR     = os.path.join(ROOT_DIR, "logs")
RESULT_DIR  = os.path.join(ROOT_DIR, "results")

# ── Data collector settings ───────────────────────────────────────────────────
COLLECT_INTERVAL_SEC  = 0.5      # how often to poll OS metrics
COLLECTOR_MAX_ROWS    = 5000     # rolling window size in memory
RAW_METRICS_CSV       = os.path.join(DATA_DIR, "raw_metrics.csv")

# ── Module 1: CPU Scheduler ───────────────────────────────────────────────────
CPU_MODEL_PATH        = os.path.join(MODEL_DIR, "cpu_scheduler_rl")
CPU_TRAIN_TIMESTEPS   = 50_000
CPU_ALGORITHMS        = ["FCFS", "SJF", "RR", "PRIORITY"]

# ── Module 2: Memory Predictor ────────────────────────────────────────────────
MEM_MODEL_PATH        = os.path.join(MODEL_DIR, "memory_lstm.pt")
MEM_SEQUENCE_LEN      = 10      # LSTM input window length
MEM_HIDDEN_SIZE       = 64
MEM_EPOCHS            = 30

# ── Module 3: Deadlock Predictor ──────────────────────────────────────────────
DEADLOCK_MODEL_PATH   = os.path.join(MODEL_DIR, "deadlock_rf.pkl")
DEADLOCK_THRESHOLD    = 0.75    # probability above this = unsafe state

# ── Module 4: Disk Optimizer ──────────────────────────────────────────────────
DISK_MODEL_PATH       = os.path.join(MODEL_DIR, "disk_kmeans.pkl")
DISK_N_CLUSTERS       = 4       # number of access pattern clusters

# ── Module 5: Anomaly Detector ────────────────────────────────────────────────
ANOMALY_MODEL_PATH    = os.path.join(MODEL_DIR, "anomaly_iforest.pkl")
ANOMALY_CONTAMINATION = 0.05    # expected 5% anomalous processes
ANOMALY_THRESHOLD     = -0.15   # isolation forest score cutoff

# ── Module 6: Energy Optimizer ────────────────────────────────────────────────
ENERGY_MODEL_PATH     = os.path.join(MODEL_DIR, "energy_rl")
ENERGY_LAMBDA         = 0.4     # weight: 0=max perf, 1=max saving
ENERGY_TRAIN_STEPS    = 40_000

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC = 1.0
DASHBOARD_PORT        = 8501

# ── bpftrace eBPF probe ───────────────────────────────────────────────────────
EBPF_OUTPUT_LOG       = os.path.join(LOG_DIR, "ebpf_sched.log")
EBPF_PROBE_SCRIPT     = os.path.join(ROOT_DIR, "utils", "sched_probe.bt")

# ── Ensure all directories exist on import ────────────────────────────────────
for _dir in [DATA_DIR, MODEL_DIR, MODULE_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(_dir, exist_ok=True)
