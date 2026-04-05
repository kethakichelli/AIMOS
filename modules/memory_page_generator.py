"""
AIMOS — Module 2: Page Access Sequence Generator
Generates realistic page access patterns using Zipf distribution
(matches real OS memory locality of reference).
Also attempts to read real /proc data on Linux.
"""

import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))
from utils.config import DATA_DIR

def zipf_sequence(n_pages=50, length=5000, alpha=1.2, seed=42):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_pages + 1)
    weights = 1.0 / (ranks ** alpha)
    weights /= weights.sum()
    return rng.choice(n_pages, size=length, p=weights).tolist()

def generate_workload_sequences(n_pages=50, seed=42):
    """Generate 4 realistic workload types."""
    rng = np.random.default_rng(seed)
    sequences = {}

    # Sequential scan (database full-table scan)
    seq = list(range(n_pages)) * 20
    sequences['sequential'] = seq[:5000]

    # Zipf locality (typical application)
    sequences['zipf'] = zipf_sequence(n_pages, 5000, alpha=1.2, seed=seed)

    # Hot-cold (80% accesses to 20% of pages)
    hot = list(range(10))
    cold = list(range(10, n_pages))
    hot_cold = []
    for _ in range(5000):
        if rng.random() < 0.8:
            hot_cold.append(int(rng.choice(hot)))
        else:
            hot_cold.append(int(rng.choice(cold)))
    sequences['hot_cold'] = hot_cold

    # Random (worst case)
    sequences['random'] = rng.integers(0, n_pages, size=5000).tolist()

    return sequences

def save_sequences(sequences, out_dir=None):
    out_dir = out_dir or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for wtype, seq in sequences.items():
        for t, page in enumerate(seq):
            all_rows.append({'workload': wtype, 'time': t, 'page_id': page})
    df = pd.DataFrame(all_rows)
    path = os.path.join(out_dir, 'page_sequences.csv')
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} page access records → {path}")
    return df

if __name__ == '__main__':
    seqs = generate_workload_sequences()
    df = save_sequences(seqs)
    print("\nSample (zipf workload, first 10):")
    print(df[df.workload=='zipf'].head(10).to_string(index=False))
    print("\nPage access distribution (zipf):")
    zipf_df = df[df.workload=='zipf']
    top = zipf_df['page_id'].value_counts().head(5)
    for page, count in top.items():
        print(f"  Page {page:2d}: {'#'*int(count/20)} ({count} accesses)")
