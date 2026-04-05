"""
AIMOS — Module 5: Security Anomaly Detector
Uses Isolation Forest to detect abnormal process behavior
in real time without any labeled malware data.

Features used per process:
  - cpu_percent       : sudden CPU spikes
  - mem_rss_mb        : memory growth
  - io_read_bytes     : abnormal read activity
  - io_write_bytes    : abnormal write activity
  - num_threads       : thread explosion
  - uptime_sec        : very new or zombie processes
  - wait_time_ns      : scheduling anomalies
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from utils.config import (
    ANOMALY_MODEL_PATH,
    ANOMALY_CONTAMINATION,
    ANOMALY_THRESHOLD,
    RAW_METRICS_CSV,
    RESULT_DIR,
    LOG_DIR
)


# ── Feature columns used by this module ──────────────────────────────────────
ANOMALY_FEATURES = [
    'cpu_percent',
    'mem_rss_mb',
    'io_read_bytes',
    'io_write_bytes',
    'num_threads',
    'uptime_sec',
    'wait_time_ns',
]


class AIMOSAnomalyDetector:
    """
    Real-time process anomaly detector for AIMOS.

    Usage:
        detector = AIMOSAnomalyDetector()
        detector.train(df)
        results = detector.predict(df)
        anomalies = results[results['is_anomaly'] == True]
    """

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.is_trained = False
        self.feature_cols = ANOMALY_FEATURES
        self.train_stats  = {}

    # ── Data preparation ──────────────────────────────────────────────────────

    def _prepare_features(self, df):
        """Extract and clean feature matrix from raw collector data."""
        proc_df = df[df['type'] == 'process'].copy()
        # Fill numeric and string columns separately (pandas 2.0+)
        num_cols = proc_df.select_dtypes(include='number').columns
        str_cols = proc_df.select_dtypes(include='object').columns
        proc_df[num_cols] = proc_df[num_cols].fillna(0)
        proc_df[str_cols] = proc_df[str_cols].fillna('')

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in proc_df.columns:
                proc_df[col] = 0.0

        # Log-scale I/O values to handle extreme outliers
        proc_df['io_read_bytes']  = np.log1p(proc_df['io_read_bytes'])
        proc_df['io_write_bytes'] = np.log1p(proc_df['io_write_bytes'])
        proc_df['wait_time_ns']   = np.log1p(proc_df['wait_time_ns'])
        proc_df['uptime_sec']     = np.log1p(proc_df['uptime_sec'])

        X = proc_df[self.feature_cols].values.astype(np.float32)
        return X, proc_df

    def _inject_synthetic_anomalies(self, df, n=30):
        """
        Inject synthetic anomalous processes for validation.
        Simulates: crypto miners, memory leaks, fork bombs, data exfil.
        """
        anomalies = []
        base_time = pd.Timestamp.now()

        for i in range(n):
            atype = np.random.choice(
                ['crypto_miner', 'memory_leak',
                 'fork_bomb', 'data_exfil'],
                p=[0.3, 0.3, 0.2, 0.2]
            )

            if atype == 'crypto_miner':
                row = {
                    'type'           : 'process',
                    'timestamp'      : base_time,
                    'pid'            : 99000 + i,
                    'name'           : f'miner_{i}',
                    'cpu_percent'    : np.random.uniform(85, 100),
                    'mem_rss_mb'     : np.random.uniform(50, 200),
                    'io_read_bytes'  : np.random.uniform(0, 1000),
                    'io_write_bytes' : np.random.uniform(0, 1000),
                    'num_threads'    : np.random.randint(4, 16),
                    'uptime_sec'     : np.random.uniform(10, 3600),
                    'wait_time_ns'   : np.random.uniform(1e6, 1e8),
                    'status'         : 'running',
                    'anomaly_type'   : 'crypto_miner',
                    'true_anomaly'   : 1,
                }
            elif atype == 'memory_leak':
                row = {
                    'type'           : 'process',
                    'timestamp'      : base_time,
                    'pid'            : 99000 + i,
                    'name'           : f'leaker_{i}',
                    'cpu_percent'    : np.random.uniform(5, 20),
                    'mem_rss_mb'     : np.random.uniform(2000, 8000),
                    'io_read_bytes'  : np.random.uniform(0, 500),
                    'io_write_bytes' : np.random.uniform(0, 500),
                    'num_threads'    : np.random.randint(1, 4),
                    'uptime_sec'     : np.random.uniform(100, 7200),
                    'wait_time_ns'   : np.random.uniform(1e7, 1e9),
                    'status'         : 'sleeping',
                    'anomaly_type'   : 'memory_leak',
                    'true_anomaly'   : 1,
                }
            elif atype == 'fork_bomb':
                row = {
                    'type'           : 'process',
                    'timestamp'      : base_time,
                    'pid'            : 99000 + i,
                    'name'           : f'forkbomb_{i}',
                    'cpu_percent'    : np.random.uniform(30, 70),
                    'mem_rss_mb'     : np.random.uniform(100, 500),
                    'io_read_bytes'  : np.random.uniform(0, 200),
                    'io_write_bytes' : np.random.uniform(0, 200),
                    'num_threads'    : np.random.randint(50, 200),
                    'uptime_sec'     : np.random.uniform(1, 30),
                    'wait_time_ns'   : np.random.uniform(1e8, 1e10),
                    'status'         : 'running',
                    'anomaly_type'   : 'fork_bomb',
                    'true_anomaly'   : 1,
                }
            else:  # data_exfil
                row = {
                    'type'           : 'process',
                    'timestamp'      : base_time,
                    'pid'            : 99000 + i,
                    'name'           : f'exfil_{i}',
                    'cpu_percent'    : np.random.uniform(10, 40),
                    'mem_rss_mb'     : np.random.uniform(50, 300),
                    'io_read_bytes'  : np.random.uniform(1e8, 1e10),
                    'io_write_bytes' : np.random.uniform(1e8, 1e10),
                    'num_threads'    : np.random.randint(2, 8),
                    'uptime_sec'     : np.random.uniform(60, 1800),
                    'wait_time_ns'   : np.random.uniform(1e6, 1e8),
                    'status'         : 'running',
                    'anomaly_type'   : 'data_exfil',
                    'true_anomaly'   : 1,
                }

            anomalies.append(row)

        anom_df = pd.DataFrame(anomalies)
        return pd.concat([df, anom_df], ignore_index=True)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df):
        """
        Train Isolation Forest on normal process behavior.
        The model learns what 'normal' looks like.
        Anomalies are anything significantly different.
        """
        print("\n[Module 5] Training Isolation Forest anomaly detector...")

        X, proc_df = self._prepare_features(df)

        if len(X) < 10:
            print("  Not enough data. Generating synthetic normal data.")
            X = self._generate_normal_data(300)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators     = 200,
            contamination    = ANOMALY_CONTAMINATION,
            max_samples      = 'auto',
            random_state= 42,
            n_jobs           = -1,
            verbose          = 0
        )
        self.model.fit(X_scaled)

        # Store training statistics for reporting
        scores = self.model.decision_function(X_scaled)
        self.train_stats = {
            'n_samples'      : len(X),
            'n_features'     : X.shape[1],
            'score_mean'     : float(np.mean(scores)),
            'score_std'      : float(np.std(scores)),
            'score_min'      : float(np.min(scores)),
            'score_max'      : float(np.max(scores)),
            'threshold'      : ANOMALY_THRESHOLD,
        }

        self.is_trained = True
        print(f"  Trained on {len(X)} process snapshots")
        print(f"  Anomaly score range: "
              f"[{self.train_stats['score_min']:.4f}, "
              f"{self.train_stats['score_max']:.4f}]")

        self._save_model()
        return self

    def _generate_normal_data(self, n=300):
        """Generate synthetic normal process behavior."""
        np.random.seed(42)
        return np.column_stack([
            np.random.exponential(2.0, n),          # cpu_percent
            np.random.exponential(50.0, n),         # mem_rss_mb
            np.random.exponential(10000, n),        # io_read_bytes
            np.random.exponential(5000, n),         # io_write_bytes
            np.random.randint(1, 8, n).astype(float), # num_threads
            np.random.exponential(500, n),          # uptime_sec
            np.random.exponential(1e7, n),          # wait_time_ns
        ])

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df):
        """
        Score all processes and flag anomalies.
        Returns DataFrame with anomaly scores and flags.
        """
        if not self.is_trained:
            raise RuntimeError("Train the model first.")

        X, proc_df = self._prepare_features(df)

        if len(X) == 0:
            return pd.DataFrame()

        X_scaled = self.scaler.transform(X)
        scores   = self.model.decision_function(X_scaled)
        preds    = self.model.predict(X_scaled)  # 1=normal, -1=anomaly

        proc_df = proc_df.copy()
        proc_df['anomaly_score'] = scores
        proc_df['is_anomaly']    = preds == -1
        proc_df['risk_level']    = proc_df['anomaly_score'].apply(
            self._score_to_risk
        )

        return proc_df[[
            'pid', 'name', 'status',
            'cpu_percent', 'mem_rss_mb',
            'anomaly_score', 'is_anomaly', 'risk_level'
        ]]

    def _score_to_risk(self, score):
        if score < -0.3:
            return 'CRITICAL'
        elif score < -0.15:
            return 'HIGH'
        elif score < 0.0:
            return 'MEDIUM'
        else:
            return 'NORMAL'

    def predict_single_process(self, process_row):
        """Score a single process dict. Used by control brain."""
        if not self.is_trained:
            return {'is_anomaly': False, 'score': 0.0, 'risk': 'NORMAL'}

        row = pd.DataFrame([process_row])
        row['type'] = 'process'
        row.fillna(0, inplace=True)

        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0.0

        row['io_read_bytes']  = np.log1p(row['io_read_bytes'])
        row['io_write_bytes'] = np.log1p(row['io_write_bytes'])
        row['wait_time_ns']   = np.log1p(row['wait_time_ns'])
        row['uptime_sec']     = np.log1p(row['uptime_sec'])

        X       = row[self.feature_cols].values.astype(np.float32)
        X_sc    = self.scaler.transform(X)
        score   = float(self.model.decision_function(X_sc)[0])
        is_anom = self.model.predict(X_sc)[0] == -1

        return {
            'is_anomaly' : is_anom,
            'score'      : score,
            'risk'       : self._score_to_risk(score)
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_with_injected_anomalies(self, df):
        """
        Inject known synthetic anomalies and measure detection rate.
        This produces your results table for the paper.
        """
        print("\n[Module 5] Evaluating with injected anomalies...")

        df_with_anom = self._inject_synthetic_anomalies(df, n=40)
        X, proc_df   = self._prepare_features(df_with_anom)
        X_scaled     = self.scaler.transform(X)

        scores = self.model.decision_function(X_scaled)
        preds  = self.model.predict(X_scaled)

        true_labels = df_with_anom[
            df_with_anom['type'] == 'process'
        ]['true_anomaly'].fillna(0).values

        pred_labels = (preds == -1).astype(int)

        # Metrics
        tp = int(np.sum((pred_labels == 1) & (true_labels == 1)))
        fp = int(np.sum((pred_labels == 1) & (true_labels == 0)))
        fn = int(np.sum((pred_labels == 0) & (true_labels == 1)))
        tn = int(np.sum((pred_labels == 0) & (true_labels == 0)))

        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1         = (2 * precision * recall /
                     (precision + recall)
                     if (precision + recall) > 0 else 0)
        accuracy   = (tp + tn) / len(true_labels)

        print(f"\n  Detection results:")
        print(f"  True Positives  : {tp}")
        print(f"  False Positives : {fp}")
        print(f"  False Negatives : {fn}")
        print(f"  True Negatives  : {tn}")
        print(f"  Precision       : {precision:.4f}")
        print(f"  Recall          : {recall:.4f}")
        print(f"  F1 Score        : {f1:.4f}")
        print(f"  Accuracy        : {accuracy:.4f}")

        # Save results
        results = {
            'metric': ['Precision','Recall','F1 Score',
                       'Accuracy','TP','FP','FN','TN'],
            'value' : [precision, recall, f1,
                       accuracy, tp, fp, fn, tn]
        }
        pd.DataFrame(results).to_csv(
            os.path.join(RESULT_DIR, 'anomaly_detection_results.csv'),
            index=False
        )

        self._plot_anomaly_scores(scores, true_labels)
        return precision, recall, f1

    def _plot_anomaly_scores(self, scores, true_labels):
        plt.figure(figsize=(10, 4))

        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]

        plt.hist(normal_scores,  bins=40, alpha=0.7,
                 color='#3B82F6', label='Normal processes')
        plt.hist(anomaly_scores, bins=20, alpha=0.7,
                 color='#EF4444', label='Anomalous processes')
        plt.axvline(ANOMALY_THRESHOLD, color='black',
                    linestyle='--', linewidth=2,
                    label=f'Threshold ({ANOMALY_THRESHOLD})')
        plt.title('AIMOS — Anomaly score distribution')
        plt.xlabel('Isolation Forest anomaly score')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()

        path = os.path.join(RESULT_DIR, 'anomaly_score_distribution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Plot saved → {path}")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _save_model(self):
        os.makedirs(os.path.dirname(ANOMALY_MODEL_PATH), exist_ok=True)
        joblib.dump({
            'model'      : self.model,
            'scaler'     : self.scaler,
            'train_stats': self.train_stats,
            'features'   : self.feature_cols,
        }, ANOMALY_MODEL_PATH)
        print(f"  Model saved → {ANOMALY_MODEL_PATH}")

    def load_model(self):
        data = joblib.load(ANOMALY_MODEL_PATH)
        self.model       = data['model']
        self.scaler      = data['scaler']
        self.train_stats = data['train_stats']
        self.feature_cols= data['features']
        self.is_trained  = True
        print(f"[Module 5] Model loaded from {ANOMALY_MODEL_PATH}")
        return self


# ── Train and evaluate ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  AIMOS — Module 5: Anomaly Detector")
    print("="*55)

    df = pd.read_csv(RAW_METRICS_CSV)
    print(f"\nLoaded {len(df)} rows from {RAW_METRICS_CSV}")

    detector = AIMOSAnomalyDetector()
    detector.train(df)

    print("\n--- Live process scoring ---")
    results = detector.predict(df)
    anomalies = results[results['is_anomaly'] == True]

    print(f"\nProcesses flagged as anomalous: {len(anomalies)}")
    if len(anomalies) > 0:
        print(anomalies[['pid','name','cpu_percent',
                         'mem_rss_mb','anomaly_score',
                         'risk_level']].to_string(index=False))

    print("\n--- Evaluation with injected anomalies ---")
    precision, recall, f1 = detector.evaluate_with_injected_anomalies(df)

    print(f"\n{'='*55}")
    print(f"  Module 5 COMPLETE")
    print(f"  Precision: {precision:.4f}  "
          f"Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"{'='*55}\n")
