import sys
import os
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

from utils.data_collector import AIMOSDataCollector
import time

if __name__ == '__main__':
    print("\n=== AIMOS Data Collector — Live Test ===\n")

    collector = AIMOSDataCollector()
    collector.start()

    print("Collecting 10 seconds of real OS data...\n")
    time.sleep(10)

    proc_df = collector.get_latest(n=100, row_type='process')
    sys_df  = collector.get_system_series(n=20)

    print(f"Total rows collected : {collector.row_count}")

    print("\n--- Top 10 processes by CPU usage ---")
    cols = ['name','cpu_percent','mem_rss_mb',
            'wait_time_ns','timeslices','status']
    print(
        proc_df[cols]
        .sort_values('cpu_percent', ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n--- System metrics (last 5 snapshots) ---")
    sys_cols = ['timestamp','cpu_percent',
                'mem_used_mb','mem_percent',
                'load_avg_1m','disk_reads']
    print(sys_df[sys_cols].tail(5).to_string(index=False))

    collector.save_to_csv()
    collector.stop()

    print("\n=== Week 1 complete. AIMOS is collecting real OS data. ===\n")
