"""
AIMOS — System Data Collector
Reads real OS metrics from psutil + /proc every 500ms.
Runs in background threads — non-blocking.
All 6 AI modules consume data from this single source.
"""

import psutil
import pandas as pd
import time
import os
import threading
import queue
import logging
from datetime import datetime
from utils.config import (
    COLLECT_INTERVAL_SEC,
    COLLECTOR_MAX_ROWS,
    RAW_METRICS_CSV,
    LOG_DIR
)

# ── Logger setup ──────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AIMOS-COLLECTOR] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'collector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIMOSDataCollector:
    """
    Core data collection engine for AIMOS.
    
    Usage:
        collector = AIMOSDataCollector()
        collector.start()
        ...
        df = collector.get_latest(n=100, row_type='process')
        sys_df = collector.get_system_series(n=200)
        collector.stop()
    """

    def __init__(self,
                 interval=COLLECT_INTERVAL_SEC,
                 max_rows=COLLECTOR_MAX_ROWS):
        self.interval   = interval
        self.max_rows   = max_rows
        self.data_queue = queue.Queue()
        self.running    = False
        self.df         = pd.DataFrame()
        self._lock      = threading.Lock()
        self._prev_cpu_total = 0
        self._prev_cpu_idle  = 0

    # ── /proc readers ─────────────────────────────────────────────────────────

    def _read_proc_meminfo(self):
        mem = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(':')] = int(parts[1])
        except Exception as e:
            logger.warning(f"/proc/meminfo error: {e}")
        return mem

    def _read_proc_diskstats(self):
        stats = {}
        try:
            with open('/proc/diskstats', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 14:
                        dev = parts[2]
                        if any(dev.startswith(d) for d in
                               ['sd', 'vd', 'nvme', 'hd', 'xvd']):
                            stats[dev] = {
                                'reads_completed' : int(parts[3]),
                                'writes_completed': int(parts[7]),
                                'sectors_read'    : int(parts[5]),
                                'sectors_written' : int(parts[9]),
                                'io_in_progress'  : int(parts[11]),
                                'ms_doing_io'     : int(parts[12]),
                            }
        except Exception as e:
            logger.warning(f"/proc/diskstats error: {e}")
        return stats

    def _read_schedstat(self, pid):
        try:
            with open(f'/proc/{pid}/schedstat', 'r') as f:
                parts = f.read().split()
                if len(parts) >= 3:
                    return {
                        'cpu_time_ns' : int(parts[0]),
                        'wait_time_ns': int(parts[1]),
                        'timeslices'  : int(parts[2]),
                    }
        except Exception:
            pass
        return {'cpu_time_ns': 0, 'wait_time_ns': 0, 'timeslices': 0}

    def _read_proc_stat_cpu(self):
        try:
            with open('/proc/stat', 'r') as f:
                parts = f.readline().split()
            vals  = [int(x) for x in parts[1:]]
            total = sum(vals)
            idle  = vals[3] + vals[4]
            delta_total = total - self._prev_cpu_total
            delta_idle  = idle  - self._prev_cpu_idle
            self._prev_cpu_total = total
            self._prev_cpu_idle  = idle
            if delta_total == 0:
                return 0.0
            return round(100.0 * (1 - delta_idle / delta_total), 2)
        except Exception:
            return 0.0

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def _collect_snapshot(self):
        ts          = datetime.now()
        cpu_pct     = psutil.cpu_percent(interval=None, percpu=True)
        cpu_proc    = self._read_proc_stat_cpu()
        mem         = psutil.virtual_memory()
        swap        = psutil.swap_memory()
        mem_proc    = self._read_proc_meminfo()
        disk        = self._read_proc_diskstats()
        load_avg    = os.getloadavg()           # 1, 5, 15 min load

        # ── System summary row ────────────────────────────────────────────────
        sys_row = {
            'timestamp'         : ts,
            'type'              : 'system',
            'pid'               : -1,
            'name'              : 'SYSTEM',

            # CPU
            'cpu_percent'       : round(sum(cpu_pct) / len(cpu_pct), 2),
            'cpu_percent_proc'  : cpu_proc,
            'cpu_count'         : len(cpu_pct),
            'load_avg_1m'       : load_avg[0],
            'load_avg_5m'       : load_avg[1],
            'load_avg_15m'      : load_avg[2],

            # Memory
            'mem_used_mb'       : round(mem.used        / 1e6, 2),
            'mem_available_mb'  : round(mem.available   / 1e6, 2),
            'mem_percent'       : mem.percent,
            'mem_cached_kb'     : mem_proc.get('Cached',   0),
            'mem_buffers_kb'    : mem_proc.get('Buffers',  0),
            'mem_dirty_kb'      : mem_proc.get('Dirty',    0),
            'mem_active_kb'     : mem_proc.get('Active',   0),
            'mem_inactive_kb'   : mem_proc.get('Inactive', 0),

            # Swap
            'swap_used_mb'      : round(swap.used / 1e6, 2),
            'swap_percent'      : swap.percent,

            # Disk (first device)
            'disk_reads'        : 0,
            'disk_writes'       : 0,
            'disk_sectors_read' : 0,
            'disk_sectors_written': 0,
            'disk_io_active'    : 0,
            'disk_ms_io'        : 0,
        }

        if disk:
            d = list(disk.values())[0]
            sys_row.update({
                'disk_reads'          : d['reads_completed'],
                'disk_writes'         : d['writes_completed'],
                'disk_sectors_read'   : d['sectors_read'],
                'disk_sectors_written': d['sectors_written'],
                'disk_io_active'      : d['io_in_progress'],
                'disk_ms_io'          : d['ms_doing_io'],
            })

        rows = [sys_row]

        # ── Per-process rows ──────────────────────────────────────────────────
        for proc in psutil.process_iter([
            'pid', 'name', 'status', 'cpu_percent',
            'memory_info', 'num_threads', 'nice',
            'io_counters', 'create_time', 'username'
        ]):
            try:
                info  = proc.info
                pid   = info['pid']
                sched = self._read_schedstat(pid)

                rows.append({
                    'timestamp'       : ts,
                    'type'            : 'process',
                    'pid'             : pid,
                    'name'            : info['name'] or 'unknown',
                    'status'          : info['status'],
                    'username'        : info.get('username', ''),
                    'cpu_percent'     : info['cpu_percent']  or 0.0,
                    'mem_rss_mb'      : round(
                        info['memory_info'].rss / 1e6, 3)
                        if info['memory_info'] else 0,
                    'mem_vms_mb'      : round(
                        info['memory_info'].vms / 1e6, 3)
                        if info['memory_info'] else 0,
                    'num_threads'     : info['num_threads']  or 1,
                    'nice'            : info['nice']         or 0,
                    'io_read_bytes'   : info['io_counters'].read_bytes
                                        if info['io_counters'] else 0,
                    'io_write_bytes'  : info['io_counters'].write_bytes
                                        if info['io_counters'] else 0,
                    # /proc kernel data
                    'cpu_time_ns'     : sched['cpu_time_ns'],
                    'wait_time_ns'    : sched['wait_time_ns'],
                    'timeslices'      : sched['timeslices'],
                    'uptime_sec'      : round(
                        time.time() - info['create_time'], 2),
                })
            except (psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue

        return rows

    # ── Background threads ────────────────────────────────────────────────────

    def _collection_loop(self):
        logger.info("AIMOS data collector started.")
        while self.running:
            try:
                self.data_queue.put(self._collect_snapshot())
            except Exception as e:
                logger.error(f"Snapshot failed: {e}")
            time.sleep(self.interval)
        logger.info("AIMOS data collector stopped.")

    def _storage_loop(self):
        while self.running or not self.data_queue.empty():
            try:
                rows = self.data_queue.get(timeout=1.0)
                new  = pd.DataFrame(rows)
                with self._lock:
                    self.df = pd.concat(
                        [self.df, new], ignore_index=True
                    ).tail(self.max_rows)
            except queue.Empty:
                continue

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start background collection threads."""
        self.running = True
        threading.Thread(
            target=self._collection_loop,
            name='AIMOS-Collector',
            daemon=True
        ).start()
        threading.Thread(
            target=self._storage_loop,
            name='AIMOS-Storage',
            daemon=True
        ).start()
        logger.info("Both threads running.")

    def stop(self):
        """Gracefully stop collection."""
        self.running = False
        logger.info("Collector stop requested.")

    def get_latest(self, n=100, row_type='process'):
        """Return last n rows of given type."""
        with self._lock:
            return self.df[
                self.df['type'] == row_type
            ].tail(n).copy()

    def get_system_series(self, n=200):
        """Return system-level time series for dashboard."""
        return self.get_latest(n=n, row_type='system')

    def get_process_by_name(self, name, n=50):
        """Return recent rows for a specific process name."""
        with self._lock:
            return self.df[
                (self.df['type'] == 'process') &
                (self.df['name'] == name)
            ].tail(n).copy()

    def save_to_csv(self, path=RAW_METRICS_CSV):
        """Save entire rolling DataFrame to CSV."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with self._lock:
            self.df.to_csv(path, index=False)
        logger.info(f"Saved {len(self.df)} rows → {path}")

    @property
    def row_count(self):
        with self._lock:
            return len(self.df)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n=== AIMOS Data Collector — Live Test ===\n")

    collector = AIMOSDataCollector()
    collector.start()

    print("Collecting 10 seconds of real OS data...\n")
    time.sleep(10)

    proc_df = collector.get_latest(n=100, row_type='process')
    sys_df  = collector.get_system_series(n=20)

    print(f"Total rows collected : {collector.row_count}")
    print(f"Process rows (last 100): {len(proc_df)}")

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

    print(f"\nSaved to: {RAW_METRICS_CSV}")
    print("=== Week 1 complete. AIMOS is collecting real OS data. ===\n")
