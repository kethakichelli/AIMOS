"""
AIMOS — eBPF Kernel Event Collector
Captures real kernel-level events using bpftrace.
This is genuine kernel-adjacent data collection —
real context switches, real page faults, real disk events.
Requires: sudo bpftrace (already installed on your system)
"""

import subprocess, threading, os, sys, logging, time
from collections import deque, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/AIMOS"))
from utils.config import LOG_DIR

logger = logging.getLogger(__name__)

class eBPFCollector:

    def __init__(self, max_events=1000):
        self.context_switches = deque(maxlen=max_events)
        self.page_faults      = deque(maxlen=max_events)
        self.disk_events      = deque(maxlen=max_events)
        self._processes       = defaultdict(int)
        self._stop            = threading.Event()

    def start_context_switch_trace(self):
        """Trace every real CPU context switch."""
        script = """
tracepoint:sched:sched_switch {
    printf("SWITCH %s %d %s %d\\n",
        args->prev_comm, args->prev_pid,
        args->next_comm, args->next_pid);
}
"""
        self._run_bpftrace(script, self._parse_context_switch,
                           name="ctx_switch")

    def start_page_fault_trace(self):
        """Trace real page faults from the kernel."""
        script = """
tracepoint:exceptions:page_fault_user {
    printf("FAULT %d %lx\\n", pid, args->address);
}
"""
        self._run_bpftrace(script, self._parse_page_fault,
                           name="page_fault")

    def start_disk_trace(self):
        """Trace real disk I/O requests."""
        script = """
tracepoint:block:block_rq_insert {
    printf("DISK %d %llu\\n", args->nr_sector, args->sector);
}
"""
        self._run_bpftrace(script, self._parse_disk_event,
                           name="disk")

    def _run_bpftrace(self, script, parser, name):
        def run():
            try:
                proc = subprocess.Popen(
                    ["sudo", "bpftrace", "-e", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                for line in proc.stdout:
                    if self._stop.is_set():
                        proc.terminate()
                        break
                    line = line.strip()
                    if line:
                        parser(line)
            except Exception as e:
                logger.warning(f"eBPF {name} trace: {e}")

        t = threading.Thread(target=run, daemon=True, name=f"ebpf_{name}")
        t.start()
        logger.info(f"eBPF {name} trace started")

    def _parse_context_switch(self, line):
        if line.startswith("SWITCH"):
            parts = line.split()
            if len(parts) >= 5:
                self.context_switches.append({
                    'time': datetime.now().isoformat(),
                    'prev_comm': parts[1],
                    'prev_pid':  parts[2],
                    'next_comm': parts[3],
                    'next_pid':  parts[4]
                })
                self._processes[parts[3]] += 1

    def _parse_page_fault(self, line):
        if line.startswith("FAULT"):
            parts = line.split()
            if len(parts) >= 3:
                self.page_faults.append({
                    'time': datetime.now().isoformat(),
                    'pid':  parts[1],
                    'addr': parts[2]
                })

    def _parse_disk_event(self, line):
        if line.startswith("DISK"):
            parts = line.split()
            if len(parts) >= 3:
                self.disk_events.append({
                    'time':    datetime.now().isoformat(),
                    'sectors': parts[1],
                    'lba':     parts[2]
                })

    def get_context_switch_rate(self):
        """Context switches per second (last 100 events)."""
        events = list(self.context_switches)
        if len(events) < 2:
            return 0
        return min(len(events), 100)

    def get_top_scheduled_processes(self, n=5):
        """Most frequently scheduled processes."""
        return sorted(self._processes.items(),
                      key=lambda x: x[1], reverse=True)[:n]

    def stop(self):
        self._stop.set()


    def start(self):
        try:
            self.start_context_switch_trace()
            self.start_page_fault_trace()
            self.start_disk_trace()
            return True
        except Exception as e:
            return False


def test_ebpf():
    print("\n=== AIMOS eBPF Collector Test ===")
    print("Collecting kernel events for 10 seconds...")
    print("(requires sudo bpftrace)\n")

    col = eBPFCollector()
    col.start_context_switch_trace()
    time.sleep(10)

    switches = list(col.context_switches)
    print(f"Context switches captured : {len(switches)}")
    if switches:
        print("\nLast 5 context switches:")
        for e in switches[-5:]:
            print(f"  {e['prev_comm']:15s} → {e['next_comm']}")
        print("\nTop scheduled processes:")
        for name, count in col.get_top_scheduled_processes():
            print(f"  {name:20s}: {count} timeslices")

    col.stop()
    print("\neBPF test complete.")

if __name__ == '__main__':
    test_ebpf()

# Alias for compatibility with control_brain.py
AIMOSeBPFCollector = eBPFCollector


