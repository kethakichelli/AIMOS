"""
AIMOS — Module 2: LSTM Page Fault Predictor
Predicts the next page(s) to be accessed using sequence learning.
Innovation: preloads predicted pages BEFORE they are requested
→ reduces page faults compared to LRU/FIFO.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/AIMOS"))
from utils.config import DATA_DIR, MODEL_DIR, MEM_SEQUENCE_LEN, MEM_HIDDEN_SIZE, MEM_EPOCHS

N_PAGES = 50

# ── Dataset ───────────────────────────────────────────────────────────────────
class PageSequenceDataset(Dataset):
    def __init__(self, sequence, seq_len=MEM_SEQUENCE_LEN):
        self.seq_len = seq_len
        self.data = sequence
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+self.seq_len],    dtype=torch.long)
        return x, y

# ── Model ─────────────────────────────────────────────────────────────────────
class PageLSTM(nn.Module):
    def __init__(self, n_pages=N_PAGES, embed_dim=32,
                 hidden=MEM_HIDDEN_SIZE, layers=2, dropout=0.2):
        super().__init__()
        self.embed   = nn.Embedding(n_pages, embed_dim)
        self.lstm    = nn.LSTM(embed_dim, hidden, layers,
                               batch_first=True, dropout=dropout)
        self.fc      = nn.Linear(hidden, n_pages)
    def forward(self, x):
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc(out[:, -1, :])

# ── Baseline algorithms ───────────────────────────────────────────────────────
def simulate_lru(sequence, cache_size=10):
    cache, faults = [], 0
    for page in sequence:
        if page not in cache:
            faults += 1
            if len(cache) >= cache_size:
                cache.pop(0)
            cache.append(page)
        else:
            cache.remove(page)
            cache.append(page)
    return faults

def simulate_fifo(sequence, cache_size=10):
    cache, faults = [], 0
    for page in sequence:
        if page not in cache:
            faults += 1
            if len(cache) >= cache_size:
                cache.pop(0)
            cache.append(page)
    return faults

def simulate_ai_prefetch(sequence, model, seq_len, cache_size=10, topk=3):
    """AI-assisted: prefetch top-k predicted pages into cache."""
    model.eval()
    cache, faults = list(range(min(cache_size, seq_len))), 0
    with torch.no_grad():
        for i in range(seq_len, len(sequence)):
            page = sequence[i]
            if page not in cache:
                faults += 1
                if len(cache) >= cache_size:
                    cache.pop(0)
                cache.append(page)
            # prefetch predictions
            window = torch.tensor(
                sequence[i-seq_len:i], dtype=torch.long).unsqueeze(0)
            logits = model(window)
            topk_pages = torch.topk(logits, topk).indices[0].tolist()
            for p in topk_pages:
                if p not in cache:
                    if len(cache) >= cache_size:
                        cache.pop(0)
                    cache.append(p)
    return faults

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.config import RESULT_DIR

    print("\n" + "="*55)
    print("  AIMOS — Module 2: LSTM Memory Predictor Training")
    print("="*55 + "\n")

    csv_path = os.path.join(DATA_DIR, 'page_sequences.csv')
    if not os.path.exists(csv_path):
        print("Generating page sequences first...")
        from modules.memory_page_generator import generate_workload_sequences, save_sequences
        save_sequences(generate_workload_sequences())

    df  = pd.read_csv(csv_path)
    seq = df[df.workload == 'zipf']['page_id'].tolist()
    print(f"Training on zipf sequence: {len(seq)} page accesses, {N_PAGES} unique pages")

    split     = int(0.8 * len(seq))
    train_seq = seq[:split]
    test_seq  = seq[split:]

    train_ds  = PageSequenceDataset(train_seq)
    train_dl  = DataLoader(train_ds, batch_size=64, shuffle=True)

    model     = PageLSTM()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    print(f"Training LSTM for {MEM_EPOCHS} epochs...")
    for epoch in range(MEM_EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        losses.append(avg)
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{MEM_EPOCHS}  loss={avg:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    from utils.config import MEM_MODEL_PATH
    torch.save(model.state_dict(), MEM_MODEL_PATH)
    print(f"\nModel saved → {MEM_MODEL_PATH}")

    # ── Evaluate: accuracy@k ──────────────────────────────────────────────────
    model.eval()
    test_ds = PageSequenceDataset(test_seq)
    hit1 = hit3 = hit5 = total = 0
    with torch.no_grad():
        for x, y in DataLoader(test_ds, batch_size=128):
            logits = model(x)
            for k, hit_var in [(1, 'hit1'), (3, 'hit3'), (5, 'hit5')]:
                preds = torch.topk(logits, k).indices
                hits  = (preds == y.unsqueeze(1)).any(dim=1).sum().item()
                if k == 1: hit1 += hits
                elif k == 3: hit3 += hits
                else: hit5 += hits
            total += len(y)

    print(f"\n--- Prediction Accuracy ---")
    print(f"  Top-1 accuracy: {hit1/total*100:.1f}%")
    print(f"  Top-3 accuracy: {hit3/total*100:.1f}%")
    print(f"  Top-5 accuracy: {hit5/total*100:.1f}%")

    # ── Page fault comparison ─────────────────────────────────────────────────
    eval_seq = test_seq[:500]
    lru_f  = simulate_lru(eval_seq)
    fifo_f = simulate_fifo(eval_seq)
    ai_f   = simulate_ai_prefetch(eval_seq, model, MEM_SEQUENCE_LEN)

    print(f"\n--- Page Fault Comparison (500 accesses, cache=10) ---")
    print(f"  FIFO faults : {fifo_f}")
    print(f"  LRU  faults : {lru_f}")
    print(f"  AI   faults : {ai_f}  ← AIMOS prediction")
    reduction = (lru_f - ai_f) / lru_f * 100
    print(f"  Reduction vs LRU: {reduction:.1f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses, color='steelblue')
    axes[0].set_title('LSTM Training Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CrossEntropy Loss')

    algos  = ['FIFO', 'LRU', 'AI (AIMOS)']
    faults = [fifo_f, lru_f, ai_f]
    colors = ['#e74c3c', '#e67e22', '#27ae60']
    bars = axes[1].bar(algos, faults, color=colors)
    axes[1].set_title('Page Faults: AI vs Traditional')
    axes[1].set_ylabel('Total Page Faults')
    for bar, val in zip(bars, faults):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.5, str(val), ha='center', fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(RESULT_DIR, 'memory_page_faults.png')
    plt.savefig(chart_path, dpi=150)
    print(f"\nChart saved → {chart_path}")

    results = {
        'top1_accuracy': round(hit1/total*100, 2),
        'top3_accuracy': round(hit3/total*100, 2),
        'top5_accuracy': round(hit5/total*100, 2),
        'fifo_faults': fifo_f, 'lru_faults': lru_f, 'ai_faults': ai_f,
        'reduction_vs_lru_pct': round(reduction, 2)
    }
    with open(os.path.join(RESULT_DIR, 'memory_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  Module 2 COMPLETE — LSTM Memory Predictor trained")
    print("="*55 + "\n")

if __name__ == '__main__':
    train()
