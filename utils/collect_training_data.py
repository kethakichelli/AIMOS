"""
AIMOS — Real Training Data Collector
Run this overnight to collect genuine system behavior data.
This replaces synthetic data with real OS patterns.
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

from utils.data_collector import AIMOSDataCollector
from utils.config import DATA_DIR
import pandas as pd
from datetime import datetime

def collect(duration_minutes=60):
    print(f"\n=== AIMOS Real Data Collector ===")
    print(f"Collecting for {duration_minutes} minutes...")
    print(f"Do normal work while this runs — browse, code, run builds.\n")

    collector = AIMOSDataCollector()
    collector.start()

    start = time.time()
    interval = 60
    chunk = 0

    while (time.time() - start) < duration_minutes * 60:
        time.sleep(interval)
        chunk += 1
        elapsed = int((time.time() - start) / 60)
        print(f"  {elapsed}/{duration_minutes} min — "
              f"{collector.row_count} rows collected")

        # Save chunk every 5 minutes
        if chunk % 5 == 0:
            path = os.path.join(
                DATA_DIR,
                f"real_data_chunk_{chunk}.csv"
            )
            df = collector.get_latest(n=5000)
            df.to_csv(path, index=False)
            print(f"  Checkpoint saved → {path}")

    collector.save_to_csv()
    collector.stop()

    # Merge all chunks
    all_files = [f for f in os.listdir(DATA_DIR)
                 if f.startswith('real_data_chunk')]
    if all_files:
        dfs = [pd.read_csv(os.path.join(DATA_DIR, f))
               for f in all_files]
        merged = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join(DATA_DIR, 'real_training_data.csv')
        merged.to_csv(merged_path, index=False)
        print(f"\nMerged {len(merged)} rows → {merged_path}")

    print(f"\n=== Collection complete. Retrain your models now. ===")

if __name__ == '__main__':
    mins = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    collect(mins)
