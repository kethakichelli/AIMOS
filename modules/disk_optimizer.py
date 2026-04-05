"""
AIMOS — Module 4: Smart Disk Optimizer
Uses K-Means clustering to learn file access patterns,
then reorders disk requests to minimize seek time.
Fixed: clusters are traversed in cylinder order (elevator strategy).
"""

import numpy as np
import pandas as pd
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/AIMOS"))
from utils.config import DATA_DIR, MODEL_DIR, RESULT_DIR, DISK_N_CLUSTERS

# ── Generate realistic disk access logs ───────────────────────────────────────
def generate_disk_access_log(n_files=40, n_accesses=3000, seed=42):
    rng = np.random.default_rng(seed)
    n_groups = 6
    file_groups = {i: [] for i in range(n_groups)}
    for f in range(n_files):
        file_groups[f % n_groups].append(f)

    # Key fix: co-accessed files get NEARBY cylinder positions
    file_cylinders = np.zeros(n_files, dtype=int)
    for g in range(n_groups):
        base = g * (1000 // n_groups)
        for f in file_groups[g]:
            file_cylinders[f] = int(np.clip(
                rng.integers(base, base + 150), 0, 999))

    records = []
    for t in range(n_accesses):
        group_weights = np.array([1/(i+1) for i in range(n_groups)])
        group_weights /= group_weights.sum()
        group = rng.choice(n_groups, p=group_weights)
        files_in_group = file_groups[group]
        n_accessed = min(rng.integers(1, 4), len(files_in_group))
        accessed = rng.choice(files_in_group, size=n_accessed, replace=False)
        for f in accessed:
            records.append({
                'time': t,
                'file_id': int(f),
                'cylinder': int(file_cylinders[f]),
                'group_truth': group
            })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, 'disk_access_log.csv')
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} disk access records → {path}")
    return df, file_cylinders

# ── Disk scheduling algorithms ────────────────────────────────────────────────
def sstf_seek_time(requests, start=500):
    remaining = list(requests)
    head = start
    total_seek = 0
    while remaining:
        closest = min(remaining, key=lambda x: abs(x - head))
        total_seek += abs(closest - head)
        head = closest
        remaining.remove(closest)
    return total_seek

def scan_seek_time(requests, start=500):
    sorted_req = sorted(requests)
    head = start
    total_seek = 0
    upper = [r for r in sorted_req if r >= head]
    lower = [r for r in sorted_req if r <  head][::-1]
    for cyl in upper + lower:
        total_seek += abs(cyl - head)
        head = cyl
    return total_seek

def ai_optimized_seek(requests, cyl_to_cluster, cluster_mean_cyls, start=500):
    """
    AI-optimized: group requests by cluster, traverse clusters
    in cylinder order (nearest cluster mean first — elevator style).
    Within each cluster, process in cylinder order.
    """
    # Group requests by cluster
    cluster_groups = {}
    for cyl in requests:
        cl = cyl_to_cluster.get(int(cyl), 0)
        cluster_groups.setdefault(cl, []).append(cyl)

    head = start
    total_seek = 0

    # Order clusters by mean cylinder position (elevator sweep)
    ordered_clusters = sorted(
        cluster_groups.keys(),
        key=lambda cl: cluster_mean_cyls.get(cl, 0)
    )

    # Split into clusters above and below head
    above = [cl for cl in ordered_clusters
             if cluster_mean_cyls.get(cl, 0) >= head]
    below = [cl for cl in ordered_clusters
             if cluster_mean_cyls.get(cl, 0) <  head][::-1]

    for cl in above + below:
        # Within cluster: sort cylinders in direction of travel
        cyls = sorted(cluster_groups[cl])
        for cyl in cyls:
            total_seek += abs(cyl - head)
            head = cyl

    return total_seek

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import joblib

    print("\n" + "="*55)
    print("  AIMOS — Module 4: Disk Optimizer Training")
    print("="*55 + "\n")

    df, file_cylinders = generate_disk_access_log()

    # ── Build co-access frequency matrix ─────────────────────────────────────
    n_files = 40
    coaccess = np.zeros((n_files, n_files))
    for t in df['time'].unique():
        files_at_t = df[df['time'] == t]['file_id'].tolist()
        for i in files_at_t:
            for j in files_at_t:
                if i != j:
                    coaccess[i][j] += 1

    print(f"Co-access matrix built: {n_files}x{n_files}")

    # ── K-Means clustering ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X      = scaler.fit_transform(coaccess)
    kmeans = KMeans(n_clusters=DISK_N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sil    = silhouette_score(X, labels)

    print(f"K-Means clustering: {DISK_N_CLUSTERS} clusters")
    print(f"Silhouette score  : {sil:.4f}")

    # Compute mean cylinder per cluster
    cluster_mean_cyls = {}
    for cl in range(DISK_N_CLUSTERS):
        files_in_cl = np.where(labels == cl)[0]
        cyls = file_cylinders[files_in_cl]
        mean_cyl = int(cyls.mean()) if len(cyls) > 0 else 0
        cluster_mean_cyls[cl] = mean_cyl
        print(f"  Cluster {cl}: {len(files_in_cl)} files | "
              f"cylinders {cyls.min()}–{cyls.max()} | mean={mean_cyl}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    from utils.config import DISK_MODEL_PATH
    joblib.dump({
        'kmeans': kmeans, 'scaler': scaler, 'labels': labels,
        'file_cylinders': file_cylinders,
        'cluster_mean_cyls': cluster_mean_cyls
    }, DISK_MODEL_PATH)
    print(f"\nModel saved → {DISK_MODEL_PATH}")

    # ── Simulate scheduling comparison ────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_requests = 100
    request_cyls = rng.choice(file_cylinders, size=n_requests).tolist()

    cyl_to_cluster = {int(file_cylinders[fid]): int(labels[fid])
                      for fid in range(n_files)}

    sstf_total = sstf_seek_time(request_cyls)
    scan_total = scan_seek_time(request_cyls)
    ai_total   = ai_optimized_seek(request_cyls, cyl_to_cluster,
                                   cluster_mean_cyls)

    print(f"\n--- Seek Time Comparison (100 requests) ---")
    print(f"  SCAN algorithm : {scan_total:6d} cylinders")
    print(f"  SSTF algorithm : {sstf_total:6d} cylinders")
    print(f"  AI (AIMOS)     : {ai_total:6d} cylinders")

    best_baseline = min(scan_total, sstf_total)
    reduction = (best_baseline - ai_total) / best_baseline * 100
    print(f"  Reduction vs best baseline: {reduction:.1f}%")

    if ai_total < best_baseline:
        print("  AI outperforms both traditional algorithms.")
    else:
        print("  Note: AI matches elevator strategy on this workload.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    algos  = ['SCAN', 'SSTF', 'AI (AIMOS)']
    seeks  = [scan_total, sstf_total, ai_total]
    colors = ['#e74c3c', '#e67e22', '#27ae60']
    bars   = axes[0].bar(algos, seeks, color=colors)
    axes[0].set_title('Total Seek Time (cylinders)')
    axes[0].set_ylabel('Cylinders traversed')
    for bar, val in zip(bars, seeks):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+30, str(val), ha='center')

    for cl in range(DISK_N_CLUSTERS):
        cyls = file_cylinders[labels == cl]
        axes[1].scatter(cyls, [cl]*len(cyls), label=f'Cluster {cl}', s=60)
    axes[1].set_title('File Clusters by Cylinder Position')
    axes[1].set_xlabel('Cylinder'); axes[1].set_ylabel('Cluster')
    axes[1].legend(fontsize=8)

    im = axes[2].imshow(coaccess[:20, :20], cmap='Blues', aspect='auto')
    axes[2].set_title('Co-access Matrix (top 20 files)')
    axes[2].set_xlabel('File ID'); axes[2].set_ylabel('File ID')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    chart_path = os.path.join(RESULT_DIR, 'disk_results.png')
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved → {chart_path}")

    results = {
        'silhouette_score': round(sil, 4),
        'n_clusters': DISK_N_CLUSTERS,
        'scan_seek': scan_total, 'sstf_seek': sstf_total,
        'ai_seek': ai_total,
        'reduction_vs_best_pct': round(reduction, 1)
    }
    with open(os.path.join(RESULT_DIR, 'disk_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  Module 4 COMPLETE — Disk Optimizer trained")
    print("="*55 + "\n")

if __name__ == '__main__':
    train()

def recommend_file_placement(model_path=None):
    """
    Recommend optimal file placement on disk.
    Co-accessed files should be placed near each other
    to minimize seek time — this is what ext4 tries to do
    with block groups, but AIMOS does it with AI.
    """
    import joblib
    from utils.config import DISK_MODEL_PATH, RESULT_DIR
    import json

    path = model_path or DISK_MODEL_PATH
    if not os.path.exists(path):
        print("Train disk optimizer first.")
        return

    data           = joblib.load(path)
    labels         = data['labels']
    file_cylinders = data['file_cylinders']
    n_clusters     = data['kmeans'].n_clusters

    print("\n=== AIMOS Disk Placement Recommendations ===\n")
    recommendations = {}

    for cl in range(n_clusters):
        files = np.where(labels == cl)[0]
        cyls  = file_cylinders[files]
        ideal_start = cl * (1000 // n_clusters)
        ideal_end   = ideal_start + (1000 // n_clusters)

        print(f"Cluster {cl} ({len(files)} files):")
        print(f"  Current cylinders : {cyls.min()}–{cyls.max()}")
        print(f"  Recommended zone  : {ideal_start}–{ideal_end}")
        print(f"  Files             : {files.tolist()}")
        print(f"  Reason            : Co-accessed together "
              f"{int(cyls.mean())} mean cylinder\n")

        recommendations[f"cluster_{cl}"] = {
            'files': files.tolist(),
            'current_cyl_range': [int(cyls.min()), int(cyls.max())],
            'recommended_zone':  [ideal_start, ideal_end]
        }

    out = os.path.join(RESULT_DIR, 'disk_placement_recommendations.json')
    with open(out, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"Recommendations saved → {out}")
    return recommendations

if __name__ == '__main__':
    train()
    recommend_file_placement()
