"""
AIMOS — Automatic Model Retrainer
Watches for new data and retrains models when enough
new samples are available. Runs as background thread.
"""

import os, sys, time, threading, logging
sys.path.insert(0, os.path.expanduser("~/AIMOS"))
import pandas as pd
import numpy as np
from utils.config import DATA_DIR, MODEL_DIR, RESULT_DIR

logger = logging.getLogger(__name__)

class ModelRetrainer:

    def __init__(self,
                 retrain_interval_min=30,
                 min_new_samples=500):
        self.interval     = retrain_interval_min * 60
        self.min_samples  = min_new_samples
        self.last_retrain = 0
        self.retrain_count = 0
        self._stop_event  = threading.Event()

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Model retrainer started — "
                    f"checking every {self.interval//60} min")

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            self._check_and_retrain()

    def _check_and_retrain(self):
        csv = os.path.join(DATA_DIR, 'raw_metrics.csv')
        if not os.path.exists(csv):
            return

        df = pd.read_csv(csv)
        new_rows = len(df) - getattr(self, '_last_row_count', 0)

        if new_rows < self.min_samples:
            logger.info(
                f"Retrainer: only {new_rows} new rows "
                f"— need {self.min_samples} to retrain")
            return

        logger.info(
            f"Retrainer: {new_rows} new rows — retraining...")
        self._last_row_count = len(df)
        self.retrain_count  += 1

        self._retrain_anomaly_detector(df)
        self._retrain_deadlock_predictor()

    def _retrain_anomaly_detector(self, df):
        """Retrain Isolation Forest on latest process data."""
        try:
            from sklearn.ensemble import IsolationForest
            import joblib

            feature_cols = [
                c for c in [
                    'cpu_percent','mem_percent',
                    'num_threads','num_fds'
                ] if c in df.columns
            ]
            if not feature_cols:
                return

            X = df[feature_cols].dropna().values
            if len(X) < 100:
                return

            model = IsolationForest(
                contamination=0.05, random_state=42, n_jobs=-1)
            model.fit(X)

            path = os.path.join(MODEL_DIR, 'anomaly_iforest.pkl')
            joblib.dump(model, path)
            logger.info(
                f"Retrained anomaly detector on "
                f"{len(X)} samples (run #{self.retrain_count})")
        except Exception as e:
            logger.error(f"Anomaly retrain failed: {e}")

    def _retrain_deadlock_predictor(self):
        """Retrain deadlock RF on freshly generated states."""
        try:
            sys.path.insert(0, os.path.expanduser("~/AIMOS"))
            from modules.deadlock_predictor import (
                generate_dataset, extract_features
            )
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            import joblib

            X, y = generate_dataset(n_samples=3000)
            scaler = StandardScaler()
            X_s    = scaler.fit_transform(X)
            rf     = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_s, y)

            path = os.path.join(MODEL_DIR, 'deadlock_rf.pkl')
            joblib.dump({'model': rf, 'scaler': scaler}, path)
            logger.info(
                f"Retrained deadlock predictor "
                f"(run #{self.retrain_count})")
        except Exception as e:
            logger.error(f"Deadlock retrain failed: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("=== AIMOS Model Retrainer — Test Run ===\n")
    retrainer = ModelRetrainer(
        retrain_interval_min=1,
        min_new_samples=10
    )
    retrainer._check_and_retrain()
    print("\nRetrainer test complete.")
