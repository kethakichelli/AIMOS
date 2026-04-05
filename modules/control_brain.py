"""
AIMOS — AI Control Brain
The central meta-controller that coordinates all 6 modules.
Runs continuously, makes unified OS decisions every second,
and resolves conflicts between competing module outputs.

This is the core innovation of AIMOS:
  Traditional OS → fixed algorithms, no adaptation
  AIMOS          → AI observes, decides, and adapts in real time
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import time
import threading
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

from utils.config import (
    LOG_DIR, RESULT_DIR,
    CPU_MODEL_PATH, ANOMALY_MODEL_PATH,
    ENERGY_MODEL_PATH, DEADLOCK_MODEL_PATH,
    DISK_MODEL_PATH, MEM_MODEL_PATH,
    CPU_ALGORITHMS, ENERGY_LAMBDA
)

os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [AIMOS-BRAIN] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'control_brain.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIMOSControlBrain:
    """
    Central intelligence layer of AIMOS.

    Responsibilities:
      1. Load all 6 trained AI models
      2. Poll the data collector every second
      3. Run all 6 modules in parallel
      4. Resolve conflicts between module decisions
      5. Produce a unified SystemDecision every cycle
      6. Log all decisions for dashboard and paper
    """

    def __init__(self, collector=None):
        self.collector   = collector
        self.running     = False
        self.cycle_count = 0
        self.start_time  = None

        # Decision history for dashboard
        self.decision_log = deque(maxlen=500)

        # Module instances — loaded lazily
        self._cpu_model      = None
        self._anomaly_model  = None
        self._energy_model   = None
        self._deadlock_model = None
        self._disk_model     = None
        self._mem_model      = None

        # Current system state
        self.current_decision = None
        self._lock = threading.Lock()

        logger.info("AIMOS Control Brain initialised.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_all_models(self):
        """Load all 6 trained AI models into memory."""
        logger.info("Loading all AI models...")
        errors = []

        # Module 1 — CPU Scheduler RL
        try:
            from stable_baselines3 import PPO
            model_path = os.path.join(CPU_MODEL_PATH, 'final_model')
            if os.path.exists(model_path + '.zip'):
                self._cpu_model = PPO.load(model_path)
                logger.info("  Module 1 CPU Scheduler    — loaded")
            else:
                logger.warning("  Module 1 CPU Scheduler    — model not found")
        except Exception as e:
            errors.append(f"CPU: {e}")

        # Module 5 — Anomaly Detector
        try:
            import joblib
            if os.path.exists(ANOMALY_MODEL_PATH):
                data = joblib.load(ANOMALY_MODEL_PATH)
                from sklearn.ensemble import IsolationForest
                self._anomaly_model  = data['model']
                self._anomaly_scaler = data['scaler']
                self._anomaly_feats  = data['features']
                logger.info("  Module 5 Anomaly Detector — loaded")
            else:
                logger.warning("  Module 5 Anomaly Detector — model not found")
        except Exception as e:
            errors.append(f"Anomaly: {e}")

        # Module 6 — Energy Optimizer
        try:
            from stable_baselines3 import PPO
            model_path = os.path.join(ENERGY_MODEL_PATH, 'final_model')
            if os.path.exists(model_path + '.zip'):
                self._energy_model = PPO.load(model_path)
                logger.info("  Module 6 Energy Optimizer — loaded")
            else:
                logger.warning("  Module 6 Energy Optimizer — model not found")
        except Exception as e:
            errors.append(f"Energy: {e}")

        # Module 3 — Deadlock Predictor
        try:
            import joblib
            if os.path.exists(DEADLOCK_MODEL_PATH):
                data = joblib.load(DEADLOCK_MODEL_PATH)
                self._deadlock_model  = data.get('model')
                self._deadlock_scaler = data.get('scaler')
                self._deadlock_feats  = data.get('features', [])
                logger.info("  Module 3 Deadlock Pred.   — loaded")
            else:
                logger.warning("  Module 3 Deadlock Pred.   — model not found")
        except Exception as e:
            errors.append(f"Deadlock: {e}")

        # Module 4 — Disk Optimizer
        try:
            import joblib
            if os.path.exists(DISK_MODEL_PATH):
                data = joblib.load(DISK_MODEL_PATH)
                self._disk_model  = data.get('model')
                self._disk_scaler = data.get('scaler')
                logger.info("  Module 4 Disk Optimizer   — loaded")
            else:
                logger.warning("  Module 4 Disk Optimizer   — model not found")
        except Exception as e:
            errors.append(f"Disk: {e}")

        # Module 2 — Memory Predictor
        try:
            mem_path = MEM_MODEL_PATH
            if os.path.exists(mem_path):
                import torch
                self._mem_model = torch.load(
                    mem_path, map_location='cpu',
                    weights_only=False
                )
                self._mem_model.eval()
                logger.info("  Module 2 Memory Predictor — loaded")
            else:
                logger.warning("  Module 2 Memory Predictor — model not found")
        except Exception as e:
            errors.append(f"Memory: {e}")

        if errors:
            logger.warning(f"Some models failed to load: {errors}")
        logger.info("Model loading complete.")

    # ── Module runners ────────────────────────────────────────────────────────

    def _run_cpu_module(self, proc_df, sys_df):
        """Module 1: Decide best scheduling algorithm."""
        try:
            if self._cpu_model is None or len(proc_df) == 0:
                return {'algorithm': 'RR', 'source': 'default'}

            cpu_vals  = proc_df['cpu_percent'].values
            wait_vals = proc_df['wait_time_ns'].values

            obs = np.array([
                np.clip(np.mean(cpu_vals) / 100.0,  0, 1),
                np.clip(np.mean(wait_vals) / 1e10,  0, 1),
                np.clip(np.std(cpu_vals)  / 50.0,   0, 1),
                float(np.mean(cpu_vals < 5.0)),
                np.clip(np.percentile(wait_vals, 75) / 1e10, 0, 1),
                np.clip(len(proc_df) / 200.0,        0, 1),
                float(np.mean(
                    proc_df['io_read_bytes'].values +
                    proc_df['io_write_bytes'].values >
                    np.median(
                        proc_df['io_read_bytes'].values +
                        proc_df['io_write_bytes'].values + 1
                    )
                )),
            ], dtype=np.float32)

            action, _ = self._cpu_model.predict(obs, deterministic=True)
            algorithm = CPU_ALGORITHMS[int(action)]
            return {'algorithm': algorithm, 'source': 'rl_model',
                    'obs': obs.tolist()}

        except Exception as e:
            logger.warning(f"CPU module error: {e}")
            return {'algorithm': 'RR', 'source': 'fallback'}

    def _run_anomaly_module(self, proc_df):
        """Module 5: Detect anomalous processes."""
        try:
            if self._anomaly_model is None or len(proc_df) == 0:
                return {'anomalies': [], 'count': 0, 'source': 'default'}

            import numpy as np
            feats = self._anomaly_feats
            for col in feats:
                if col not in proc_df.columns:
                    proc_df[col] = 0.0

            pdf = proc_df.copy()
            pdf['io_read_bytes']  = np.log1p(pdf['io_read_bytes'])
            pdf['io_write_bytes'] = np.log1p(pdf['io_write_bytes'])
            pdf['wait_time_ns']   = np.log1p(pdf['wait_time_ns'])
            pdf['uptime_sec']     = np.log1p(pdf['uptime_sec'])

            X      = pdf[feats].fillna(0).values.astype(np.float32)
            X_sc   = self._anomaly_scaler.transform(X)
            scores = self._anomaly_model.decision_function(X_sc)
            preds  = self._anomaly_model.predict(X_sc)

            anomalies = []
            for i, (pred, score) in enumerate(zip(preds, scores)):
                if pred == -1:
                    row = proc_df.iloc[i]
                    anomalies.append({
                        'pid'   : int(row.get('pid',  0)),
                        'name'  : str(row.get('name', 'unknown')),
                        'score' : float(score),
                        'cpu'   : float(row.get('cpu_percent', 0)),
                        'mem'   : float(row.get('mem_rss_mb',  0)),
                    })

            return {
                'anomalies' : anomalies,
                'count'     : len(anomalies),
                'source'    : 'isolation_forest'
            }

        except Exception as e:
            logger.warning(f"Anomaly module error: {e}")
            return {'anomalies': [], 'count': 0, 'source': 'fallback'}

    def _run_energy_module(self, sys_df):
        """Module 6: Select CPU governor."""
        try:
            if self._energy_model is None or len(sys_df) == 0:
                return {'governor': 'balanced', 'source': 'default'}

            row = sys_df.iloc[-1]
            obs = np.array([
                np.clip(row.get('cpu_percent',    0) / 100.0, 0, 1),
                np.clip(row.get('mem_percent',    0) / 100.0, 0, 1),
                np.clip(row.get('load_avg_1m',    0) / 8.0,   0, 1),
                np.clip(row.get('disk_io_active', 0) / 10.0,  0, 1),
                np.clip(row.get('swap_percent',   0) / 100.0, 0, 1),
                float(datetime.now().hour / 23.0),
            ], dtype=np.float32)

            from modules.energy_optimizer import ENERGY_GOVERNORS
            action, _ = self._energy_model.predict(obs, deterministic=True)
            governor  = ENERGY_GOVERNORS[int(action)]
            return {'governor': governor, 'source': 'rl_model'}

        except Exception as e:
            logger.warning(f"Energy module error: {e}")
            return {'governor': 'balanced', 'source': 'fallback'}

    def _run_deadlock_module(self, proc_df):
        """Module 3: Predict deadlock risk."""
        try:
            if self._deadlock_model is None or len(proc_df) == 0:
                return {'risk': 'LOW', 'probability': 0.0,
                        'source': 'default'}

            feats = self._deadlock_feats
            if not feats:
                return {'risk': 'LOW', 'probability': 0.0,
                        'source': 'no_features'}

            for col in feats:
                if col not in proc_df.columns:
                    proc_df[col] = 0.0

            X     = proc_df[feats].fillna(0).values.astype(np.float32)
            X_sc  = self._deadlock_scaler.transform(X)
            proba = self._deadlock_model.predict_proba(X_sc)[:, 1]
            max_p = float(np.max(proba))

            risk = ('HIGH'   if max_p > 0.7 else
                    'MEDIUM' if max_p > 0.4 else 'LOW')

            return {'risk': risk, 'probability': max_p,
                    'source': 'random_forest'}

        except Exception as e:
            logger.warning(f"Deadlock module error: {e}")
            return {'risk': 'LOW', 'probability': 0.0,
                    'source': 'fallback'}

    def _run_memory_module(self, proc_df, sys_df):
        """Module 2: Predict memory pressure."""
        try:
            if len(sys_df) < 2:
                return {'pressure': 'LOW', 'predicted_usage': 0.0,
                        'source': 'insufficient_data'}

            current_mem = float(sys_df.iloc[-1].get('mem_percent', 0))
            trend = (sys_df['mem_percent'].tail(5).diff().mean()
                     if len(sys_df) >= 5 else 0)

            pressure = ('HIGH'   if current_mem > 80 or trend > 2 else
                        'MEDIUM' if current_mem > 60 or trend > 0.5
                        else 'LOW')

            return {
                'pressure'        : pressure,
                'current_usage'   : current_mem,
                'trend_per_cycle' : float(trend),
                'source'          : 'trend_analysis'
            }

        except Exception as e:
            logger.warning(f"Memory module error: {e}")
            return {'pressure': 'LOW', 'predicted_usage': 0.0,
                    'source': 'fallback'}

    def _run_disk_module(self, proc_df, sys_df):
        """Module 4: Determine disk access pattern."""
        try:
            if len(sys_df) == 0:
                return {'pattern': 'sequential', 'source': 'default'}

            row         = sys_df.iloc[-1]
            disk_reads  = float(row.get('disk_reads',  0))
            disk_writes = float(row.get('disk_writes', 0))
            io_active   = float(row.get('disk_io_active', 0))

            total = disk_reads + disk_writes
            if total == 0:
                pattern = 'idle'
            elif disk_reads / (total + 1) > 0.7:
                pattern = 'read_heavy'
            elif disk_writes / (total + 1) > 0.7:
                pattern = 'write_heavy'
            else:
                pattern = 'mixed'

            return {
                'pattern'   : pattern,
                'io_active' : io_active,
                'source'    : 'disk_analyzer'
            }

        except Exception as e:
            logger.warning(f"Disk module error: {e}")
            return {'pattern': 'sequential', 'source': 'fallback'}

    # ── Conflict resolution ───────────────────────────────────────────────────

    def _resolve_conflicts(self, cpu_dec, energy_dec,
                           anomaly_dec, deadlock_dec,
                           mem_dec, disk_dec):
        """
        Resolve conflicts between module decisions.
        This is the meta-controller logic — the core innovation.

        Rules:
          1. Security always wins — anomaly overrides performance
          2. Deadlock risk overrides normal scheduling
          3. Memory pressure adjusts energy governor
          4. Energy governor respects CPU load decision
        """
        final = {
            'cpu_algorithm' : cpu_dec['algorithm'],
            'energy_governor': energy_dec['governor'],
            'alerts'        : [],
            'overrides'     : [],
        }

        # Rule 1: Security override
        if anomaly_dec['count'] > 0:
            final['alerts'].append(
                f"{anomaly_dec['count']} anomalous process(es) detected"
            )
            # Switch to Priority scheduling to deprioritize
            # suspicious processes
            if cpu_dec['algorithm'] != 'PRIORITY':
                final['cpu_algorithm'] = 'PRIORITY'
                final['overrides'].append(
                    'CPU→PRIORITY (security override)'
                )

        # Rule 2: Deadlock override
        if deadlock_dec['risk'] == 'HIGH':
            final['alerts'].append(
                f"Deadlock risk HIGH "
                f"(p={deadlock_dec['probability']:.2f})"
            )
            final['cpu_algorithm'] = 'RR'
            final['overrides'].append('CPU→RR (deadlock prevention)')

        # Rule 3: Memory pressure adjusts energy
        if mem_dec['pressure'] == 'HIGH':
            final['alerts'].append("Memory pressure HIGH")
            if final['energy_governor'] == 'powersave':
                final['energy_governor'] = 'balanced'
                final['overrides'].append(
                    'Energy→balanced (memory pressure)'
                )

        # Rule 4: Heavy load overrides powersave
        if (cpu_dec['algorithm'] == 'PRIORITY' and
                final['energy_governor'] == 'powersave'):
            final['energy_governor'] = 'performance'
            final['overrides'].append(
                'Energy→performance (heavy load)'
            )

        final['deadlock_risk']   = deadlock_dec['risk']
        final['memory_pressure'] = mem_dec['pressure']
        final['disk_pattern']    = disk_dec['pattern']
        final['anomaly_count']   = anomaly_dec['count']
        final['anomaly_list']    = anomaly_dec['anomalies']

        return final

    # ── Main decision cycle ───────────────────────────────────────────────────

    def _run_cycle(self):
        """One full decision cycle — runs every second."""
        try:
            # Get latest data from collector
            if self.collector:
                proc_df = self.collector.get_latest(
                    n=100, row_type='process')
                sys_df  = self.collector.get_system_series(n=20)
            else:
                # Standalone mode — read from CSV
                df      = pd.read_csv(
                    os.path.join(
                        os.path.expanduser("~/AIMOS"),
                        "data", "raw_metrics.csv"
                    )
                )
                proc_df = df[df['type'] == 'process'].tail(100)
                sys_df  = df[df['type'] == 'system'].tail(20)

            if len(proc_df) == 0:
                return

            # Run all 6 modules
            cpu_dec      = self._run_cpu_module(proc_df, sys_df)
            anomaly_dec  = self._run_anomaly_module(proc_df.copy())
            energy_dec   = self._run_energy_module(sys_df)
            deadlock_dec = self._run_deadlock_module(proc_df.copy())
            mem_dec      = self._run_memory_module(proc_df, sys_df)
            disk_dec     = self._run_disk_module(proc_df, sys_df)

            # Resolve conflicts → unified decision
            decision = self._resolve_conflicts(
                cpu_dec, energy_dec, anomaly_dec,
                deadlock_dec, mem_dec, disk_dec
            )

            # Add metadata
            decision['timestamp']   = datetime.now().isoformat()
            decision['cycle']       = self.cycle_count
            decision['cpu_percent'] = float(
                sys_df.iloc[-1].get('cpu_percent', 0)
                if len(sys_df) > 0 else 0
            )
            decision['mem_percent'] = float(
                sys_df.iloc[-1].get('mem_percent', 0)
                if len(sys_df) > 0 else 0
            )
            decision['process_count'] = len(proc_df['pid'].unique())

            # Store decision
            with self._lock:
                self.current_decision = decision
                self.decision_log.append(decision)

            self.cycle_count += 1

            # Log to console every 5 cycles
            if self.cycle_count % 5 == 0:
                self._log_decision(decision)

        except Exception as e:
            logger.error(f"Cycle error: {e}")

    def _log_decision(self, d):
        alerts   = ', '.join(d['alerts'])   or 'none'
        override = ', '.join(d['overrides'])or 'none'
        logger.info(
            f"Cycle {d['cycle']:04d} | "
            f"CPU: {d['cpu_percent']:5.1f}% | "
            f"Algo: {d['cpu_algorithm']:<8} | "
            f"Governor: {d['energy_governor']:<11} | "
            f"Deadlock: {d['deadlock_risk']:<6} | "
            f"Anomalies: {d['anomaly_count']} | "
            f"Alerts: {alerts}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, interval=1.0):
        """Start the control brain loop."""
        self.running    = True
        self.start_time = datetime.now()
        logger.info("AIMOS Control Brain started.")

        def loop():
            while self.running:
                self._run_cycle()
                time.sleep(interval)

        self._thread = threading.Thread(
            target=loop, name='AIMOS-Brain', daemon=True
        )
        self._thread.start()

    def stop(self):
        self.running = False
        logger.info(
            f"Control Brain stopped after "
            f"{self.cycle_count} cycles."
        )

    def get_current_decision(self):
        with self._lock:
            return dict(self.current_decision) \
                   if self.current_decision else {}

    def get_decision_history(self):
        with self._lock:
            return list(self.decision_log)

    def save_decision_log(self):
        history = self.get_decision_history()
        if not history:
            return
        rows = []
        for d in history:
            rows.append({
                'timestamp'      : d.get('timestamp'),
                'cycle'          : d.get('cycle'),
                'cpu_algorithm'  : d.get('cpu_algorithm'),
                'energy_governor': d.get('energy_governor'),
                'deadlock_risk'  : d.get('deadlock_risk'),
                'memory_pressure': d.get('memory_pressure'),
                'disk_pattern'   : d.get('disk_pattern'),
                'anomaly_count'  : d.get('anomaly_count'),
                'cpu_percent'    : d.get('cpu_percent'),
                'mem_percent'    : d.get('mem_percent'),
                'alerts'         : str(d.get('alerts', [])),
                'overrides'      : str(d.get('overrides', [])),
            })
        path = os.path.join(RESULT_DIR, 'control_brain_log.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Decision log saved → {path}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  AIMOS — AI Control Brain")
    print("="*55 + "\n")

    brain = AIMOSControlBrain()
    brain.load_all_models()

    print("\nRunning 10 decision cycles...\n")
    brain.start(interval=1.0)
    time.sleep(12)
    brain.stop()

    print("\n--- Final system decision ---")
    d = brain.get_current_decision()
    if d:
        print(f"  CPU Algorithm   : {d.get('cpu_algorithm')}")
        print(f"  Energy Governor : {d.get('energy_governor')}")
        print(f"  Deadlock Risk   : {d.get('deadlock_risk')}")
        print(f"  Memory Pressure : {d.get('memory_pressure')}")
        print(f"  Disk Pattern    : {d.get('disk_pattern')}")
        print(f"  Anomalies Found : {d.get('anomaly_count')}")
        print(f"  Active Alerts   : {d.get('alerts')}")
        print(f"  Overrides Applied:{d.get('overrides')}")

    brain.save_decision_log()

    print(f"\n{'='*55}")
    print(f"  Control Brain COMPLETE — {brain.cycle_count} cycles run")
    print(f"{'='*55}\n")


class AIMOSControlBrainWithEBPF(AIMOSControlBrain):
    """
    Enhanced Control Brain that uses eBPF telemetry
    for Module 1 CPU scheduling decisions.
    Extends the base brain with kernel-level observations.
    """

    def __init__(self, collector=None):
        super().__init__(collector)
        from utils.ebpf_collector import AIMOSeBPFCollector
        self._ebpf = AIMOSeBPFCollector()
        self._ebpf_active = False

    def start_with_ebpf(self, interval=1.0):
        """Start brain with eBPF probe attached."""
        logger.info("Starting eBPF probe...")
        self._ebpf_active = self._ebpf.start()

        if self._ebpf_active:
            logger.info(
                "eBPF active — Module 1 using kernel telemetry"
            )
        else:
            logger.warning(
                "eBPF unavailable — Module 1 using /proc fallback"
            )

        self.start(interval)

    def _run_cpu_module(self, proc_df, sys_df):
        """Override: use eBPF observation if available."""
        if self._ebpf_active and self._cpu_model:
            try:
                obs = self._ebpf.get_rl_observation()
                action, _ = self._cpu_model.predict(
                    obs, deterministic=True
                )
                algorithm = CPU_ALGORITHMS[int(action)]
                summary   = self._ebpf.get_stats_summary()
                return {
                    'algorithm'       : algorithm,
                    'source'          : 'ebpf_kernel_telemetry',
                    'switches_per_sec': summary['switches_per_sec'],
                    'obs'             : obs.tolist(),
                }
            except Exception as e:
                logger.warning(f"eBPF CPU module error: {e}")

        # Fall back to /proc-based observation
        return super()._run_cpu_module(proc_df, sys_df)

    def stop(self):
        if self._ebpf_active:
            self._ebpf.stop()
        super().stop()


class AIMOSControlBrainFull(AIMOSControlBrainWithEBPF):
    """
    Complete AIMOS Control Brain with full enforcement.
    Combines all 6 AI modules + eBPF + meta-controller
    + kernel enforcement via renice/ionice/cpufreq.
    This is the complete system as described in the paper.
    """

    def __init__(self, collector=None):
        super().__init__(collector)
        from modules.kernel_enforcer import AIMOSKernelEnforcer
        self._enforcer = AIMOSKernelEnforcer()
        logger.info("Kernel enforcer attached to Control Brain.")

    def _run_cycle(self):
        super()._run_cycle()
        decision = self.get_current_decision()
        if not decision:
            return
        try:
            if self.collector:
                proc_df = self.collector.get_latest(
                    n=50, row_type='process'
                )
            else:
                df = pd.read_csv(
                    os.path.join(
                        os.path.expanduser("~/AIMOS"),
                        "data", "raw_metrics.csv"
                    )
                )
                proc_df = df[
                    df['type'] == 'process'
                ].tail(50)

            enforcement = self._enforcer.enforce_decision(
                decision, proc_df
            )

            with self._lock:
                if self.current_decision:
                    self.current_decision['enforcement'] = {
                        'cpu_calls' : len(
                            enforcement.get('cpu', [])
                        ),
                        'disk_calls': len(
                            enforcement.get('disk', [])
                        ),
                        'energy_ok' : enforcement.get(
                            'energy', False
                        ),
                        'isolated'  : len(
                            enforcement.get('isolation', [])
                        ),
                        'total'     : enforcement.get(
                            'total_calls', 0
                        ),
                    }
        except Exception as e:
            logger.error(f"Enforcement cycle error: {e}")

    def stop(self):
        super().stop()
        self._enforcer.save_log()
        summary = self._enforcer.get_enforcement_summary()
        logger.info(
            f"Total enforcement: "
            f"{summary.get('total_kernel_calls', 0)} "
            f"kernel calls across "
            f"{summary.get('total_cycles', 0)} cycles"
        )
