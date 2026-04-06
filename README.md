# AIMOS — Adaptive Intelligent Management of OS

> An AI-driven OS management layer that replaces static kernel
> algorithms with self-learning models that adapt to real workload
> behaviour in real time.

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-orange)]()
[![eBPF](https://img.shields.io/badge/eBPF-bpftrace%200.20-green)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## What is AIMOS

Traditional operating systems follow fixed algorithms written
years ago. Round Robin always gives equal CPU time. Banker's
Algorithm always checks deadlock the same way. These rules
never change regardless of what the system is actually doing.

AIMOS introduces an AI Control Brain that sits between
applications and the kernel. It observes real OS behaviour
every 500ms via psutil, /proc, and eBPF kernel tracepoints,
runs 6 AI models in parallel, resolves conflicts between them,
and writes decisions back to the kernel via renice, ionice,
and cpufreq — all in real time.

---

## Architecture
---

## System Flowchart
```mermaid
flowchart TD
    A([START]) --> B

    subgraph DC["DATA COLLECTION LAYER"]
        B1[psutil
CPU · RAM · Processes]
        B2[/proc filesystem
schedstat · meminfo]
        B3[eBPF bpftrace
sched_switch tracepoint]
    end

    B --> B1 & B2 & B3

    B1 & B2 & B3 --> C

    subgraph FE["FEATURE ENGINEERING"]
        C[Normalize → Window → Encode
7-element observation vector]
    end

    C --> D

    subgraph AI["6 AI MODULES — running in parallel"]
        D --> M1 & M2 & M3 & M4 & M5 & M6
        M1["M1 CPU Scheduler
PPO RL Agent
FCFS · SJF · RR · Priority"]
        M2["M2 Memory Manager
LSTM
Page fault prediction"]
        M3["M3 Deadlock Predictor
Random Forest
Safe · Unsafe states"]
        M4["M4 Disk Optimizer
K-Means
Access pattern clustering"]
        M5["M5 Anomaly Detector
Isolation Forest
Zero-day detection"]
        M6["M6 Energy Optimizer
Multi-objective RL
Pareto-optimal governor"]
    end

    M1 & M2 & M3 & M4 & M5 & M6 --> CB

    subgraph BRAIN["AI CONTROL BRAIN — meta-controller"]
        CB[Collect all 6 outputs] --> CF{Conflict
detected?}
        CF -->|Security alert| OV1[Override → PRIORITY scheduling]
        CF -->|Deadlock HIGH| OV2[Override → Round Robin]
        CF -->|Memory HIGH| OV3[Override → balanced governor]
        CF -->|No conflict| UD[Unified decision produced]
        OV1 & OV2 & OV3 --> UD
    end

    UD --> KE

    subgraph ENFORCE["KERNEL ENFORCEMENT LAYER"]
        KE[Apply decision to OS]
        KE --> R1[renice
CPU priority]
        KE --> R2[ionice
I/O class]
        KE --> R3[cpufreq
Frequency governor]
    end

    R1 & R2 & R3 --> K

    subgraph KERNEL["LINUX KERNEL"]
        K[CPU Scheduler · Memory Manager · I/O Scheduler]
    end

    K -->|feedback loop every 1s| C

    style DC fill:#1e3a5f,color:#fff,stroke:#3B82F6
    style FE fill:#1e3a2f,color:#fff,stroke:#10B981
    style AI fill:#2d1b4e,color:#fff,stroke:#8B5CF6
    style BRAIN fill:#3d2010,color:#fff,stroke:#F59E0B
    style ENFORCE fill:#3d1010,color:#fff,stroke:#EF4444
    style KERNEL fill:#1a1a2e,color:#fff,stroke:#6B7280
```

## 6 AI Modules

| # | Module | Algorithm | OS Concept | Innovation |
|---|--------|-----------|------------|------------|
| 1 | CPU Scheduler | PPO (RL) | FCFS, SJF, RR, Priority | Selects best algorithm per workload dynamically |
| 2 | Memory Manager | LSTM | Paging, page faults | Predicts next page access before fault occurs |
| 3 | Deadlock Predictor | Random Forest | Banker's Algorithm | Detects unsafe states earlier than Banker's |
| 4 | Disk Optimizer | K-Means | Disk scheduling, seek time | Clusters access patterns to reduce seek time |
| 5 | Anomaly Detector | Isolation Forest | System calls, process states | Zero-day detection without signatures |
| 6 | Energy Optimizer | Multi-objective RL | CPU freq scaling | Balances performance vs power on Pareto front |

---

## Results

### Module 1 — CPU Scheduler (PPO Reinforcement Learning)

| Algorithm | Avg Reward | Avg Wait Time |
|-----------|-----------|---------------|
| FCFS      | 58.32     | 0.000734      |
| SJF       | 70.02     | 0.000734      |
| **RR (AIMOS)** | **81.72** | **0.000734** |
| PRIORITY  | -23.58    | 0.000734      |

AIMOS RL agent selects RR for interactive workloads and
PRIORITY under security threats — 39% higher reward than
static FCFS baseline.

---

### Module 2 — Memory Predictor (LSTM)

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 27.07% |
| Top-3 accuracy | 49.49% |
| Top-5 accuracy | 58.28% |
| FIFO page faults | 219 |
| LRU page faults | 176 |
| **AIMOS page faults** | **168** |
| **Reduction vs LRU** | **4.55%** |

LSTM predicts the next memory page before it is requested,
reducing page faults below LRU — the best traditional algorithm.

---

### Module 3 — Deadlock Predictor (Random Forest)

| Metric | Value |
|--------|-------|
| **RF ROC-AUC** | **0.9939** |
| SVM ROC-AUC | 0.9913 |
| Banker's Algorithm (ms) | 0.026 |
| AIMOS prediction (ms) | 0.178 |

ROC-AUC of 0.9939 means the classifier correctly identifies
99.4% of unsafe states. Operates proactively — flags danger
before deadlock conditions are fully met.

---

### Module 4 — Disk Optimizer (K-Means)

| Metric | Value |
|--------|-------|
| Silhouette score | 0.3762 |
| Clusters found | 4 |
| SCAN seek distance | 1414 |
| SSTF seek distance | 1414 |
| Access patterns identified | sequential, random, mixed, idle |

4 distinct disk access patterns identified and classified.
Cluster-aware I/O reordering applied via ionice.

---

### Module 5 — Anomaly Detector (Isolation Forest)

| Metric | Value |
|--------|-------|
| **Recall** | **1.0000** |
| Precision | 0.2260 |
| F1 Score | 0.3687 |
| Accuracy | 0.9504 |
| True Positives | 40 |
| False Negatives | 0 |
| True Negatives | 2585 |

Recall of 1.0 means zero missed anomalies — every injected
crypto miner, fork bomb, memory leak, and data exfiltration
process was detected. No malware signatures required.

---

### Module 6 — Energy Optimizer (Multi-objective RL)

| Governor | Avg Reward |
|----------|-----------|
| performance (fixed) | -60.18 |
| balanced (fixed) | -6.50 |
| powersave (fixed) | -2.08 |
| **AIMOS-RL** | **+30.38** |

AIMOS-RL achieves +30.38 reward vs -60.18 for fixed
performance mode — learns the Pareto-optimal trade-off
between CPU performance and power consumption.

---

## Data Collection — 5 Layers---

## Kernel Enforcement

AIMOS does not just recommend — it enforces decisions:

| Interface | Status | What it controls |
|-----------|--------|-----------------|
| renice | AVAILABLE | Process CPU priority (-20 to 19) |
| ionice | AVAILABLE | Process I/O scheduling class |
| cpufreq | WSL2-limited | CPU frequency governor |
| cgroups v2 | WSL2-limited | Hard CPU/memory limits |

47+ real kernel calls per 5 cycles measured on WSL2 Ubuntu.
Full cpufreq and cgroups enforcement active on native Linux.

---

## Setup
```bash
git clone https://github.com/kethakichelli/AIMOS.git
cd AIMOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python utils/verify_setup.py
```

### Verify eBPF works
```bash
sudo apt install -y bpftrace
sudo bpftrace -e 'BEGIN { printf("eBPF OK\n"); exit(); }'
```

---

## Run
```bash
# Step 1 — collect live OS data
python run_collector.py

# Step 2 — train all modules
python modules/train_cpu_scheduler.py
python modules/retrain_cpu_ebpf.py
python modules/memory_lstm.py
python modules/deadlock_predictor.py
python modules/disk_optimizer.py
python modules/anomaly_detector.py
python modules/energy_optimizer.py

# Step 3 — run full system
python run_aimos_full.py

# Step 4 — launch dashboard
streamlit run dashboard/app.py

# Step 5 — run all results
python run_all_results.py
```

---

## Project Structure
AIMOS/
├── utils/
│   ├── config.py              # Central configuration
│   ├── data_collector.py      # psutil + /proc collector
│   ├── ebpf_collector.py      # eBPF kernel telemetry
│   └── sched_probe.bt         # bpftrace probe script
├── modules/
│   ├── cpu_scheduler_env.py   # Custom Gym environment
│   ├── cpu_scheduler_env_ebpf.py  # eBPF-trained env
│   ├── train_cpu_scheduler.py # Module 1 training
│   ├── retrain_cpu_ebpf.py    # Module 1 eBPF retrain
│   ├── memory_lstm.py         # Module 2 LSTM
│   ├── deadlock_predictor.py  # Module 3 Random Forest
│   ├── disk_optimizer.py      # Module 4 K-Means
│   ├── anomaly_detector.py    # Module 5 Isolation Forest
│   ├── energy_optimizer.py    # Module 6 Multi-obj RL
│   ├── control_brain.py       # AI meta-controller
│   └── kernel_enforcer.py     # Kernel write-back layer
├── dashboard/
│   └── app.py                 # Streamlit live dashboard
├── results/                   # All evaluation outputs
├── run_collector.py           # Data collection runner
├── run_aimos_full.py          # Full system runner
├── run_all_results.py         # Complete results runner
└── requirements.txt

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| RL Framework | stable-baselines3, gymnasium |
| Deep Learning | PyTorch |
| ML | scikit-learn |
| Kernel Interface | psutil, /proc, bpftrace, eBPF |
| Dashboard | Streamlit, Plotly |
| OS | Ubuntu 22.04 / WSL2 |

---

## Novel Contribution

The AIMOS Control Brain is a meta-controller that runs all
6 AI models simultaneously and resolves conflicts between
their outputs using weighted priority rules based on current
system state. No existing OS implements this unified
AI-driven control layer.

Example conflict resolved automatically:
- Module 6 recommends powersave (low CPU load)
- Module 5 detects anomalous process simultaneously
- Control Brain overrides to PRIORITY scheduling
- Anomalous process deprioritized via renice

This conflict resolution is the core academic contribution.

---

## Academic Context

This project demonstrates learning-augmented OS algorithms —
an active research direction at OSDI, SOSP, and EuroSys.
Individual components mirror production systems at scale:
- Google Borg: ML-based container scheduling
- Meta Twine: unified resource management
- Netflix: eBPF-based system observability
- CrowdStrike: behavioural anomaly detection

AIMOS integrates all these concepts into one unified system.

---

## License

MIT License — free to use for research and education.
