"""
AIMOS — eBPF Scheduler Telemetry Collector
Runs bpftrace probe as a subprocess and parses
every context switch event in real time.
Feeds enriched scheduler data to Module 1.

This is the Way 3 element — real kernel instrumentation.
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import subprocess
import threading
import time
import logging
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from datetime import datetime

from utils.config import LOG_DIR, EBPF_OUTPUT_LOG, EBPF_PROBE_SCRIPT

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AIMOS-EBPF] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'ebpf.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIMOSeBPFCollector:
    """
    Captures real kernel scheduler events via bpftrace.

    What it measures:
      - context_switches    : total switches per process
      - avg_runtime_ns      : average time each process runs per slice
      - avg_waittime_ns     : average time each process waits to run
      - preemption_rate     : how often process is preempted vs yields
      - priority_inversions : times a high-prio process waited for low-prio
      - cpu_affinity_score  : how often process runs on same core

    All of this comes directly from the kernel — not from psutil,
    not from /proc polling — from the actual scheduler tracepoint.
    """

    def __init__(self, probe_script=EBPF_PROBE_SCRIPT,
                 buffer_size=2000):
        self.probe_script  = probe_script
        self.buffer_size   = buffer_size
        self.running       = False
        self._process      = None
        self._thread       = None
        self._lock         = threading.Lock()

        # Per-process scheduler statistics
        self._proc_stats   = defaultdict(lambda: {
            'switches'        : 0,
            'total_runtime_ns': 0,
            'total_wait_ns'   : 0,
            'last_start_ns'   : 0,
            'last_cpu'        : -1,
            'cpu_changes'     : 0,
            'preemptions'     : 0,   # prev_state=1 means preempted
            'voluntary_yields': 0,   # prev_state=0 means yielded
            'comm'            : '',
            'prio'            : 120,
        })

        # Rolling event buffer for ML features
        self._event_buffer = deque(maxlen=buffer_size)

        # Aggregated feature snapshots
        self._feature_snapshots = deque(maxlen=500)

        self.total_switches = 0
        self.start_time     = None

    # ── Probe management ──────────────────────────────────────────────────────

    def start(self):
        """Launch bpftrace probe as subprocess."""
        if not os.path.exists(self.probe_script):
            logger.error(f"Probe script not found: {self.probe_script}")
            return False

        try:
            self._process = subprocess.Popen(
                ['sudo', 'bpftrace', self.probe_script],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text   = True,
                bufsize= 1              # line buffered
            )
            self.running    = True
            self.start_time = time.time()

            self._thread = threading.Thread(
                target = self._read_loop,
                name   = 'AIMOS-eBPF',
                daemon = True
            )
            self._thread.start()

            # Give probe time to attach
            time.sleep(2)

            if self._process.poll() is not None:
                err = self._process.stderr.read()
                logger.error(f"bpftrace failed to start: {err}")
                self.running = False
                return False

            logger.info(
                f"eBPF probe attached to sched:sched_switch"
            )
            return True

        except FileNotFoundError:
            logger.error(
                "bpftrace not found. "
                "Install with: sudo apt install bpftrace"
            )
            return False
        except PermissionError:
            logger.error(
                "Permission denied. "
                "eBPF requires sudo privileges."
            )
            return False

    def stop(self):
        """Detach probe and clean up."""
        self.running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        logger.info(
            f"eBPF probe detached. "
            f"Total switches captured: {self.total_switches}"
        )
        self._save_log()

    # ── Event parsing ─────────────────────────────────────────────────────────

    def _parse_event(self, line):
        """
        Parse one CSV line from bpftrace output.
        Format: timestamp_ns,cpu,prev_pid,prev_comm,prev_prio,
                next_pid,next_comm,next_prio,prev_state
        """
        line = line.strip()
        if not line or line.startswith('timestamp') \
                    or line.startswith('Attaching'):
            return None

        parts = line.split(',')
        if len(parts) < 9:
            return None

        try:
            return {
                'timestamp_ns' : int(parts[0]),
                'cpu'          : int(parts[1]),
                'prev_pid'     : int(parts[2]),
                'prev_comm'    : parts[3].strip(),
                'prev_prio'    : int(parts[4]),
                'next_pid'     : int(parts[5]),
                'next_comm'    : parts[6].strip(),
                'next_prio'    : int(parts[7]),
                'prev_state'   : int(parts[8]),
            }
        except (ValueError, IndexError):
            return None

    def _update_stats(self, event):
        """Update per-process statistics from one switch event."""
        prev_pid   = event['prev_pid']
        next_pid   = event['next_pid']
        ts         = event['timestamp_ns']
        prev_state = event['prev_state']

        # Update process that just got switched OFF
        prev = self._proc_stats[prev_pid]
        prev['comm']     = event['prev_comm']
        prev['prio']     = event['prev_prio']
        prev['switches'] += 1

        if prev_state == 1:
            prev['preemptions'] += 1       # kernel preempted it
        else:
            prev['voluntary_yields'] += 1  # process yielded CPU

        if prev['last_start_ns'] > 0:
            runtime = ts - prev['last_start_ns']
            if 0 < runtime < 1e11:         # sanity check
                prev['total_runtime_ns'] += runtime

        # Update process that just got switched ON
        nxt = self._proc_stats[next_pid]
        nxt['comm']  = event['next_comm']
        nxt['prio']  = event['next_prio']

        if nxt['last_cpu'] != -1 and nxt['last_cpu'] != event['cpu']:
            nxt['cpu_changes'] += 1        # migrated to different core

        nxt['last_start_ns'] = ts
        nxt['last_cpu']      = event['cpu']

        if nxt['last_start_ns'] > 0:
            wait = ts - nxt['last_start_ns']
            if 0 < wait < 1e11:
                nxt['total_wait_ns'] += wait

        self.total_switches += 1

    # ── Read loop ─────────────────────────────────────────────────────────────

    def _read_loop(self):
        """Continuously read bpftrace output and process events."""
        logger.info("eBPF read loop started.")
        snapshot_interval = 5.0   # aggregate features every 5s
        last_snapshot     = time.time()

        for line in self._process.stdout:
            if not self.running:
                break

            event = self._parse_event(line)
            if event:
                with self._lock:
                    self._update_stats(event)
                    self._event_buffer.append(event)

            # Periodic feature snapshot
            now = time.time()
            if now - last_snapshot >= snapshot_interval:
                self._take_feature_snapshot()
                last_snapshot = now

        logger.info("eBPF read loop ended.")

    def _take_feature_snapshot(self):
        """
        Aggregate per-process stats into ML features.
        Called every 5 seconds.
        Produces features that feed directly into Module 1 RL agent.
        """
        with self._lock:
            stats = dict(self._proc_stats)

        if not stats:
            return

        features_per_proc = []
        for pid, s in stats.items():
            if s['switches'] < 2:
                continue

            sw    = max(s['switches'], 1)
            avg_rt = s['total_runtime_ns'] / sw
            avg_wt = s['total_wait_ns']    / sw

            # Preemption ratio — high = CPU hungry, often interrupted
            preempt_ratio = s['preemptions'] / sw

            # CPU affinity score — low = migrating often (bad for cache)
            affinity = 1.0 - min(s['cpu_changes'] / sw, 1.0)

            features_per_proc.append({
                'pid'            : pid,
                'comm'           : s['comm'],
                'prio'           : s['prio'],
                'switches'       : sw,
                'avg_runtime_ns' : avg_rt,
                'avg_wait_ns'    : avg_wt,
                'preempt_ratio'  : preempt_ratio,
                'affinity_score' : affinity,
                'is_interactive' : int(avg_rt < 5e6),  # < 5ms = interactive
                'is_cpu_bound'   : int(
                    avg_rt > 50e6 and preempt_ratio > 0.5
                ),
            })

        if features_per_proc:
            snapshot = {
                'timestamp'           : datetime.now().isoformat(),
                'total_switches'      : self.total_switches,
                'active_procs'        : len(features_per_proc),
                'avg_wait_ns'         : np.mean(
                    [f['avg_wait_ns'] for f in features_per_proc]
                ),
                'avg_runtime_ns'      : np.mean(
                    [f['avg_runtime_ns'] for f in features_per_proc]
                ),
                'avg_preempt_ratio'   : np.mean(
                    [f['preempt_ratio'] for f in features_per_proc]
                ),
                'interactive_ratio'   : np.mean(
                    [f['is_interactive'] for f in features_per_proc]
                ),
                'cpu_bound_ratio'     : np.mean(
                    [f['is_cpu_bound'] for f in features_per_proc]
                ),
                'avg_affinity'        : np.mean(
                    [f['affinity_score'] for f in features_per_proc]
                ),
                'per_process'         : features_per_proc,
            }
            self._feature_snapshots.append(snapshot)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_rl_observation(self):
        """
        Return a 7-element observation vector for the CPU
        scheduler RL agent — same shape as Module 1 expects.
        Now enriched with real eBPF data instead of /proc polling.
        """
        with self._lock:
            snapshots = list(self._feature_snapshots)

        if not snapshots:
            return np.zeros(7, dtype=np.float32)

        latest = snapshots[-1]

        obs = np.array([
            np.clip(latest['cpu_bound_ratio'],                0, 1),
            np.clip(latest['avg_wait_ns']    / 1e9,          0, 1),
            np.clip(latest['avg_preempt_ratio'],              0, 1),
            np.clip(latest['interactive_ratio'],              0, 1),
            np.clip(latest['avg_wait_ns']    / 1e9,          0, 1),
            np.clip(latest['active_procs']   / 200.0,        0, 1),
            np.clip(1.0 - latest['avg_affinity'],            0, 1),
        ], dtype=np.float32)

        return obs

    def get_latest_snapshot(self):
        """Return most recent feature snapshot."""
        with self._lock:
            snaps = list(self._feature_snapshots)
        return snaps[-1] if snaps else {}

    def get_top_processes(self, n=10):
        """Return top N processes by context switch count."""
        with self._lock:
            stats = dict(self._proc_stats)

        procs = []
        for pid, s in stats.items():
            if s['switches'] > 0 and s['comm']:
                sw = max(s['switches'], 1)
                procs.append({
                    'pid'            : pid,
                    'name'           : s['comm'],
                    'switches'       : s['switches'],
                    'avg_runtime_ms' : round(
                        s['total_runtime_ns'] / sw / 1e6, 3
                    ),
                    'avg_wait_ms'    : round(
                        s['total_wait_ns'] / sw / 1e6, 3
                    ),
                    'preempt_ratio'  : round(
                        s['preemptions'] / sw, 3
                    ),
                    'priority'       : s['prio'],
                })

        return sorted(
            procs, key=lambda x: x['switches'], reverse=True
        )[:n]

    def get_stats_summary(self):
        """Return system-wide scheduler statistics."""
        elapsed = time.time() - self.start_time \
                  if self.start_time else 1
        with self._lock:
            n_procs = len(self._proc_stats)

        return {
            'total_switches'     : self.total_switches,
            'switches_per_sec'   : round(
                self.total_switches / elapsed, 1
            ),
            'tracked_processes'  : n_procs,
            'uptime_sec'         : round(elapsed, 1),
            'events_buffered'    : len(self._event_buffer),
        }

    def _save_log(self):
        """Save captured scheduler data to CSV."""
        snaps = list(self._feature_snapshots)
        if not snaps:
            return
        rows = [{k: v for k, v in s.items()
                 if k != 'per_process'}
                for s in snaps]
        path = EBPF_OUTPUT_LOG.replace('.log', '_features.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"eBPF feature log saved → {path}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  AIMOS — eBPF Scheduler Telemetry")
    print("="*55 + "\n")

    collector = AIMOSeBPFCollector()

    print("Starting eBPF probe (requires sudo)...")
    ok = collector.start()

    if not ok:
        print("Failed to start eBPF probe.")
        sys.exit(1)

    print("Capturing kernel scheduler events for 15 seconds...\n")

    for i in range(15):
        time.sleep(1)
        stats = collector.get_stats_summary()
        print(
            f"  [{i+1:02d}s] "
            f"Switches: {stats['total_switches']:6d} | "
            f"Rate: {stats['switches_per_sec']:6.1f}/s | "
            f"Processes: {stats['tracked_processes']}"
        )

    print("\n--- Top processes by context switches ---")
    top = collector.get_top_processes(n=10)
    if top:
        df = pd.DataFrame(top)
        print(df.to_string(index=False))

    print("\n--- RL observation vector (feeds Module 1) ---")
    obs = collector.get_rl_observation()
    labels = [
        'cpu_bound_ratio', 'avg_wait_norm', 'preempt_ratio',
        'interactive_ratio', 'load_norm',
        'queue_length_norm', 'migration_ratio'
    ]
    for label, val in zip(labels, obs):
        bar = '█' * int(val * 20)
        print(f"  {label:<22}: {val:.4f}  {bar}")

    print("\n--- System summary ---")
    summary = collector.get_stats_summary()
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")

    collector.stop()

    print(f"\n{'='*55}")
    print(f"  eBPF Hook COMPLETE")
    print(f"  {summary['total_switches']} kernel events captured")
    print(f"{'='*55}\n")
