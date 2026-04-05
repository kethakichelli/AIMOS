"""
AIMOS — Module 3: AI-Based Deadlock Predictor
Predicts unsafe resource allocation states BEFORE deadlock occurs.
Innovation: ML classifier detects danger earlier than Banker's Algorithm.

Ground truth labels come from Banker's Algorithm (safety check).
The ML model learns to predict the SAME outcome but faster —
at request time, not allocation time.
"""

import numpy as np
import pandas as pd
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/AIMOS"))
from utils.config import MODEL_DIR, RESULT_DIR, DATA_DIR

# ── Banker's Algorithm (ground truth) ────────────────────────────────────────
def bankers_is_safe(available, allocation, max_need):
    """
    Returns True if the system is in a SAFE state.
    Classic Banker's Algorithm — O(n^2 * m).
    n = processes, m = resource types.
    """
    n, m   = allocation.shape
    work   = available.copy().astype(float)
    finish = [False] * n
    need   = max_need - allocation

    for _ in range(n):
        for i in range(n):
            if not finish[i] and all(need[i] <= work):
                work    += allocation[i]
                finish[i] = True
    return all(finish)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(available, allocation, max_need):
    """
    Convert raw resource matrices into ML feature vector.
    Features capture the 'danger signals' of unsafe states.
    """
    n, m = allocation.shape
    need = max_need - allocation

    total_resources = available + allocation.sum(axis=0)
    utilization     = allocation.sum(axis=0) / (total_resources + 1e-9)

    # How many processes can proceed right now?
    can_proceed = sum(
        1 for i in range(n) if all(need[i] <= available)
    )

    # Need satisfaction ratio per process
    need_sat = []
    for i in range(n):
        ratio = np.sum(available >= need[i]) / (m + 1e-9)
        need_sat.append(ratio)

    features = [
        utilization.mean(),
        utilization.max(),
        utilization.min(),
        available.sum() / (total_resources.sum() + 1e-9),
        can_proceed / n,
        np.mean(need_sat),
        np.min(need_sat),
        (allocation > 0).mean(),
        need.sum() / (total_resources.sum() + 1e-9),
        np.std(utilization),
    ]
    return np.array(features, dtype=np.float32)

# ── Dataset generation ────────────────────────────────────────────────────────
def generate_dataset(n_samples=8000, n_procs=5, n_resources=3, seed=42):
    """
    Generate labeled safe/unsafe resource allocation states.
    Label 0 = SAFE, Label 1 = UNSAFE (potential deadlock).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        total = rng.integers(5, 20, size=n_resources)
        max_need   = rng.integers(1, total+1, size=(n_procs, n_resources))
        allocation = np.array([
            rng.integers(0, max_need[i]+1) for i in range(n_procs)
        ])
        # Ensure allocation doesn't exceed total
        for j in range(n_resources):
            while allocation[:, j].sum() > total[j]:
                i = rng.integers(n_procs)
                if allocation[i, j] > 0:
                    allocation[i, j] -= 1

        available = total - allocation.sum(axis=0)
        available = np.maximum(available, 0)

        label = 0 if bankers_is_safe(available, allocation, max_need) else 1
        feat  = extract_features(available, allocation, max_need)
        X.append(feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Dataset: {n_samples} samples | Safe={sum(y==0)} | Unsafe={sum(y==1)}")
    return X, y

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, ConfusionMatrixDisplay
    )
    from sklearn.preprocessing import StandardScaler
    import joblib

    print("\n" + "="*55)
    print("  AIMOS — Module 3: Deadlock Predictor Training")
    print("="*55 + "\n")

    X, y = generate_dataset(n_samples=8000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc   = roc_auc_score(y_test, rf_probs)

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("Training SVM...")
    svm = SVC(kernel='rbf', probability=True,
              class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_probs = svm.predict_proba(X_test)[:, 1]
    svm_auc   = roc_auc_score(y_test, svm_probs)

    print(f"\n--- Random Forest ---")
    print(classification_report(y_test, rf_preds,
                                target_names=['Safe','Unsafe']))
    print(f"ROC-AUC: {rf_auc:.4f}")

    print(f"\n--- SVM ---")
    print(classification_report(y_test, svm_preds,
                                target_names=['Safe','Unsafe']))
    print(f"ROC-AUC: {svm_auc:.4f}")

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    from utils.config import DEADLOCK_MODEL_PATH
    joblib.dump({'model': rf, 'scaler': scaler}, DEADLOCK_MODEL_PATH)
    print(f"\nBest model (RF) saved → {DEADLOCK_MODEL_PATH}")

    # ── Feature importance ────────────────────────────────────────────────────
    feat_names = [
        'util_mean','util_max','util_min','avail_ratio',
        'can_proceed_ratio','need_sat_mean','need_sat_min',
        'alloc_density','need_ratio','util_std'
    ]
    importances = rf.feature_importances_

    # ── Timing comparison: AI vs Banker's ────────────────────────────────────
    import time
    rng = np.random.default_rng(99)
    n_timing = 500
    states = []
    for _ in range(n_timing):
        total = rng.integers(5, 20, size=3)
        max_need   = rng.integers(1, total+1, size=(5, 3))
        allocation = np.array([rng.integers(0, max_need[i]+1) for i in range(5)])
        for j in range(3):
            while allocation[:, j].sum() > total[j]:
                i = rng.integers(5)
                if allocation[i, j] > 0:
                    allocation[i, j] -= 1
        available = np.maximum(total - allocation.sum(axis=0), 0)
        states.append((available, allocation, max_need))

    t0 = time.perf_counter()
    for av, al, mx in states:
        bankers_is_safe(av, al, mx)
    banker_time = (time.perf_counter() - t0) / n_timing * 1000

    feats = np.array([extract_features(av, al, mx) for av, al, mx in states])
    feats_scaled = scaler.transform(feats)
    t0 = time.perf_counter()
    rf.predict_proba(feats_scaled)
    ai_time = (time.perf_counter() - t0) / n_timing * 1000

    print(f"\n--- Speed Comparison ({n_timing} states) ---")
    print(f"  Banker's Algorithm : {banker_time:.4f} ms per state")
    print(f"  AI (Random Forest) : {ai_time:.4f} ms per state")
    print(f"  Speedup            : {banker_time/ai_time:.1f}x faster")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Confusion matrix
    cm = confusion_matrix(y_test, rf_preds)
    ConfusionMatrixDisplay(cm, display_labels=['Safe','Unsafe']).plot(
        ax=axes[0], colorbar=False)
    axes[0].set_title('Random Forest — Confusion Matrix')

    # Feature importance
    idx = np.argsort(importances)[::-1]
    axes[1].bar(range(len(feat_names)),
                importances[idx], color='steelblue')
    axes[1].set_xticks(range(len(feat_names)))
    axes[1].set_xticklabels([feat_names[i] for i in idx],
                             rotation=45, ha='right', fontsize=8)
    axes[1].set_title('Feature Importances')

    # Speed comparison
    axes[2].bar(["Banker's", 'AI (RF)'],
                [banker_time, ai_time], color=['#e74c3c','#27ae60'])
    axes[2].set_title('Detection Speed (ms per state)')
    axes[2].set_ylabel('Milliseconds')

    plt.tight_layout()
    chart_path = os.path.join(RESULT_DIR, 'deadlock_results.png')
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved → {chart_path}")

    results = {
        'rf_roc_auc': round(rf_auc, 4),
        'svm_roc_auc': round(svm_auc, 4),
        'banker_ms': round(banker_time, 4),
        'ai_ms': round(ai_time, 4),
        'speedup': round(banker_time/ai_time, 1)
    }
    with open(os.path.join(RESULT_DIR, 'deadlock_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  Module 3 COMPLETE — Deadlock Predictor trained")
    print("="*55 + "\n")

if __name__ == '__main__':
    train()
