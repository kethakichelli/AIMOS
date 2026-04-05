# AIMOS — AI-Driven Autonomous Intelligent Management of OS

An intelligent OS management layer that uses multiple AI models
(RL, LSTM, Random Forest, K-Means, Isolation Forest) to monitor,
predict, and optimize system resources in real time.

## Architecture
- Reads real kernel data via /proc, psutil, eBPF
- 6 AI modules running in parallel
- Streamlit dashboard for live visualization
- Control Brain orchestrating all decisions

## Modules
| Module | AI Used | What it does |
|--------|---------|--------------|
| CPU Scheduler | PPO (RL) | Selects best scheduling algorithm dynamically |
| Memory Manager | LSTM | Predicts page faults before they happen |
| Deadlock Predictor | Random Forest | Detects unsafe states earlier than Banker's |
| Disk Optimizer | K-Means | Reduces seek time via co-access clustering |
| Anomaly Detector | Isolation Forest | Zero-day process anomaly detection |
| Energy Optimizer | Multi-obj RL | Balances performance vs power vs temperature |

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/AIMOS.git
cd AIMOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python utils/verify_setup.py
```

## Run
```bash
# Collect live OS data
python run_collector.py

# Train any module
python modules/train_cpu_scheduler.py
python modules/memory_lstm.py
python modules/deadlock_predictor.py
python modules/disk_optimizer.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Results
| Module | Metric | Result |
|--------|--------|--------|
| CPU Scheduler | Best algorithm selected | RR (81.7 reward) |
| Memory Predictor | Page fault reduction vs LRU | 4.5% |
| Deadlock Predictor | ROC-AUC | >0.95 |
| Disk Optimizer | Seek time reduction | >15% |

## Tech Stack
Python 3.12, PyTorch, scikit-learn, stable-baselines3,
gymnasium, Streamlit, Plotly, psutil, eBPF/bpftrace

## Project Type
Hybrid AI-OS layer — real /proc + psutil kernel interfaces
feeding live data to AI models. Same architecture as
Google Autopilot and Meta Twine.
