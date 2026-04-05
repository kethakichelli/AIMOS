"""
AIMOS — Kernel Enforcement Layer
Closes Layer 4 gap — writes AI decisions back to the OS.

Enforcement interfaces:
  renice      — changes process CPU priority (nice value)
  ionice      — changes process I/O scheduling class
  cpufreq     — changes CPU frequency governor
  cgroups v2  — changes process CPU/memory limits

All are standard Linux APIs used by every major cloud provider.
No kernel modification needed — pure userspace control.
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import subprocess
import logging
import time
import psutil
import pandas as pd
from datetime import datetime
from utils.config import LOG_DIR, RESULT_DIR

os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [AIMOS-ENFORCER] %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, 'enforcer.log')
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIMOSKernelEnforcer:
    """
    Translates AI decisions into actual OS kernel calls.

    What each interface controls:
      renice   : process CPU priority (-20 highest, 19 lowest)
      ionice   : process I/O class (realtime/best-effort/idle)
      cpufreq  : CPU frequency governor (performance/powersave etc)
      cgroups  : hard CPU and memory limits per process group
    """

    def __init__(self):
        self.enforcement_log = []
        self.capabilities    = self._detect_capabilities()
        self._log_capabilities()

    # ── Capability detection ──────────────────────────────────────────────────

    def _detect_capabilities(self):
        """
        Detect which enforcement interfaces are available.
        WSL2 has some restrictions vs native Linux.
        """
        caps = {
            'renice'  : False,
            'ionice'  : False,
            'cpufreq' : False,
            'cgroups' : False,
        }

        # renice — available if we can run as root
        try:
            r = subprocess.run(
                ['which', 'renice'],
                capture_output=True, text=True
            )
            caps['renice'] = r.returncode == 0
        except Exception:
            pass

        # ionice — check if binary exists
        try:
            r = subprocess.run(
                ['which', 'ionice'],
                capture_output=True, text=True
            )
            caps['ionice'] = r.returncode == 0
        except Exception:
            pass

        # cpufreq — check if sysfs interface exists
        cpufreq_path = (
            '/sys/devices/system/cpu/cpu0'
            '/cpufreq/scaling_governor'
        )
        caps['cpufreq'] = os.path.exists(cpufreq_path)

        # cgroups v2 — check if unified hierarchy mounted
        cgroup_path = '/sys/fs/cgroup/cgroup.controllers'
        caps['cgroups'] = os.path.exists(cgroup_path)

        return caps

    def _log_capabilities(self):
        logger.info("Kernel enforcement capabilities:")
        for cap, available in self.capabilities.items():
            status = "AVAILABLE" if available else "LIMITED"
            logger.info(f"  {cap:<10}: {status}")

    # ── renice — CPU priority enforcement ────────────────────────────────────

    def enforce_cpu_algorithm(self, algorithm, proc_df):
        """
        Enforce CPU scheduling decision via renice.

        Algorithm → nice value mapping:
          FCFS     → nice  0  (normal priority, FIFO-like)
          SJF      → nice -5  (slightly elevated for short jobs)
          RR       → nice  0  (normal, kernel handles RR)
          PRIORITY → nice varies by process importance
        """
        if not self.capabilities['renice']:
            logger.warning("renice not available")
            return []

        enforced = []

        try:
            if algorithm == 'PRIORITY':
                enforced = self._apply_priority_scheduling(
                    proc_df
                )
            elif algorithm == 'SJF':
                enforced = self._apply_sjf_scheduling(proc_df)
            elif algorithm in ['FCFS', 'RR']:
                enforced = self._apply_normal_scheduling(
                    proc_df
                )
        except Exception as e:
            logger.error(f"CPU enforcement error: {e}")

        return enforced

    def _apply_priority_scheduling(self, proc_df):
        """
        PRIORITY mode: high CPU processes get lower nice value
        (higher priority). Anomalous processes get higher nice
        value (lower priority — deprioritized).
        """
        enforced = []
        if len(proc_df) == 0:
            return enforced

        # Sort by CPU usage
        sorted_df = proc_df.sort_values(
            'cpu_percent', ascending=False
        )

        for i, (_, row) in enumerate(sorted_df.head(10).iterrows()):
            pid = int(row.get('pid', 0))
            if pid <= 1 or pid == os.getpid():
                continue

            # Top CPU users get priority boost
            if i < 3:
                nice_val = -5
            elif i < 6:
                nice_val = 0
            else:
                nice_val = 5

            result = self._renice(pid, nice_val)
            if result:
                enforced.append({
                    'pid'      : pid,
                    'name'     : row.get('name', 'unknown'),
                    'nice'     : nice_val,
                    'reason'   : 'priority_scheduling',
                })

        return enforced

    def _apply_sjf_scheduling(self, proc_df):
        """
        SJF mode: processes with shorter CPU bursts
        (lower cpu_time_ns) get slight priority boost.
        """
        enforced = []
        if len(proc_df) == 0:
            return enforced

        # Sort by CPU time — shorter burst = higher priority
        if 'cpu_time_ns' in proc_df.columns:
            sorted_df = proc_df.sort_values('cpu_time_ns')
        else:
            sorted_df = proc_df

        for i, (_, row) in enumerate(
                sorted_df.head(8).iterrows()):
            pid = int(row.get('pid', 0))
            if pid <= 1 or pid == os.getpid():
                continue

            nice_val = -3 if i < 4 else 3
            result = self._renice(pid, nice_val)
            if result:
                enforced.append({
                    'pid'    : pid,
                    'name'   : row.get('name', 'unknown'),
                    'nice'   : nice_val,
                    'reason' : 'sjf_scheduling',
                })

        return enforced

    def _apply_normal_scheduling(self, proc_df):
        """
        FCFS/RR mode: reset all processes to nice 0.
        Let kernel handle fair scheduling.
        """
        enforced = []
        for _, row in proc_df.head(10).iterrows():
            pid = int(row.get('pid', 0))
            if pid <= 1 or pid == os.getpid():
                continue
            current_nice = row.get('nice', 0)
            if current_nice != 0:
                result = self._renice(pid, 0)
                if result:
                    enforced.append({
                        'pid'    : pid,
                        'name'   : row.get('name', 'unknown'),
                        'nice'   : 0,
                        'reason' : 'reset_to_normal',
                    })
        return enforced

    def _renice(self, pid, nice_value):
        """Apply nice value to a process via renice command."""
        try:
            # Validate process still exists
            if not psutil.pid_exists(pid):
                return False

            result = subprocess.run(
                ['renice', '-n', str(nice_value),
                 '-p', str(pid)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.debug(
                    f"renice pid={pid} nice={nice_value} OK"
                )
                return True
            else:
                logger.debug(
                    f"renice pid={pid} failed: "
                    f"{result.stderr.strip()}"
                )
                return False
        except Exception as e:
            logger.debug(f"renice error pid={pid}: {e}")
            return False

    # ── ionice — I/O priority enforcement ────────────────────────────────────

    def enforce_disk_pattern(self, pattern, proc_df):
        """
        Enforce disk optimization via ionice.

        Pattern → I/O class mapping:
          read_heavy  → best-effort class 2 (normal reads)
          write_heavy → best-effort class 7 (background writes)
          mixed       → best-effort class 4 (balanced)
          idle        → idle class (lowest priority I/O)
        """
        if not self.capabilities['ionice']:
            logger.warning("ionice not available")
            return []

        class_map = {
            'read_heavy' : ('2', '2'),  # best-effort, high prio
            'write_heavy': ('2', '7'),  # best-effort, low prio
            'mixed'      : ('2', '4'),  # best-effort, medium
            'idle'       : ('3', '0'),  # idle class
        }

        io_class, io_level = class_map.get(
            pattern, ('2', '4')
        )
        enforced = []

        # Apply to top I/O processes
        if 'io_read_bytes' in proc_df.columns:
            io_col = 'io_read_bytes'
        elif 'io_write_bytes' in proc_df.columns:
            io_col = 'io_write_bytes'
        else:
            return enforced

        sorted_df = proc_df.sort_values(
            io_col, ascending=False
        )

        for _, row in sorted_df.head(5).iterrows():
            pid = int(row.get('pid', 0))
            if pid <= 1 or pid == os.getpid():
                continue

            result = self._ionice(pid, io_class, io_level)
            if result:
                enforced.append({
                    'pid'      : pid,
                    'name'     : row.get('name', 'unknown'),
                    'io_class' : io_class,
                    'io_level' : io_level,
                    'reason'   : f'disk_{pattern}',
                })

        return enforced

    def _ionice(self, pid, io_class, io_level):
        """Apply I/O scheduling class to a process."""
        try:
            if not psutil.pid_exists(pid):
                return False

            result = subprocess.run(
                ['ionice', '-c', io_class,
                 '-n', io_level, '-p', str(pid)],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"ionice error pid={pid}: {e}")
            return False

    # ── cpufreq — CPU frequency enforcement ──────────────────────────────────

    def enforce_energy_governor(self, governor):
        """
        Write CPU frequency governor to kernel sysfs.
        Works on native Linux. WSL2 has limited cpufreq access.

        Governor mapping:
          performance → maximum frequency always
          balanced    → ondemand (kernel scales dynamically)
          powersave   → minimum frequency
          adaptive    → schedutil (scheduler-driven scaling)
        """
        if not self.capabilities['cpufreq']:
            # WSL2 fallback — use process priorities instead
            logger.info(
                f"cpufreq not available (WSL2) — "
                f"governor '{governor}' logged only"
            )
            return False

        # Map AIMOS governor names to Linux governor names
        governor_map = {
            'performance': 'performance',
            'balanced'   : 'ondemand',
            'powersave'  : 'powersave',
            'adaptive'   : 'schedutil',
        }
        linux_governor = governor_map.get(
            governor, 'ondemand'
        )

        cpu_count = psutil.cpu_count()
        success_count = 0

        for cpu_id in range(cpu_count):
            gov_path = (
                f'/sys/devices/system/cpu/cpu{cpu_id}'
                f'/cpufreq/scaling_governor'
            )
            try:
                with open(gov_path, 'w') as f:
                    f.write(linux_governor)
                success_count += 1
            except PermissionError:
                logger.warning(
                    f"Permission denied: {gov_path}"
                )
                break
            except Exception as e:
                logger.warning(f"cpufreq error: {e}")
                break

        if success_count > 0:
            logger.info(
                f"CPU governor → {linux_governor} "
                f"({success_count}/{cpu_count} cores)"
            )
            return True
        return False

    def read_current_governor(self):
        """Read current CPU frequency governor from kernel."""
        gov_path = (
            '/sys/devices/system/cpu/cpu0'
            '/cpufreq/scaling_governor'
        )
        try:
            with open(gov_path, 'r') as f:
                return f.read().strip()
        except Exception:
            return 'unknown (WSL2)'

    # ── cgroups v2 — resource limits ─────────────────────────────────────────

    def enforce_anomaly_isolation(self, anomaly_list):
        """
        Isolate anomalous processes using available methods.
        On native Linux: cgroups v2 CPU throttling.
        On WSL2: renice to 19 (lowest priority).
        """
        if not anomaly_list:
            return []

        enforced = []
        for anom in anomaly_list:
            pid  = anom.get('pid', 0)
            name = anom.get('name', 'unknown')

            if pid <= 1 or pid == os.getpid():
                continue

            if not psutil.pid_exists(pid):
                continue

            if self.capabilities['cgroups']:
                result = self._cgroup_throttle(pid, name)
            else:
                # WSL2 fallback: lowest nice value
                result = self._renice(pid, 19)

            if result:
                enforced.append({
                    'pid'    : pid,
                    'name'   : name,
                    'action' : 'isolated',
                    'method' : 'cgroups'
                              if self.capabilities['cgroups']
                              else 'renice_19',
                    'reason' : 'anomaly_detected',
                })
                logger.warning(
                    f"ISOLATED: {name} (pid={pid}) — "
                    f"anomaly score: {anom.get('score',0):.4f}"
                )

        return enforced

    def _cgroup_throttle(self, pid, name):
        """Throttle process CPU via cgroups v2."""
        try:
            cgroup_name = f"aimos_isolated_{pid}"
            cgroup_path = f"/sys/fs/cgroup/{cgroup_name}"

            os.makedirs(cgroup_path, exist_ok=True)

            # Limit to 10% CPU (100000 period, 10000 quota)
            with open(
                f"{cgroup_path}/cpu.max", 'w'
            ) as f:
                f.write("10000 100000")

            # Add process to cgroup
            with open(
                f"{cgroup_path}/cgroup.procs", 'w'
            ) as f:
                f.write(str(pid))

            logger.info(
                f"cgroup throttle applied: "
                f"{name} (pid={pid}) → 10% CPU max"
            )
            return True
        except Exception as e:
            logger.debug(f"cgroup error: {e}")
            # Fall back to renice
            return self._renice(pid, 19)

    # ── Unified enforcement ───────────────────────────────────────────────────

    def enforce_decision(self, decision, proc_df):
        """
        Main entry point — enforce a complete Control Brain
        decision across all available kernel interfaces.
        Called every cycle by the Control Brain.
        """
        timestamp  = datetime.now().isoformat()
        results    = {
            'timestamp'  : timestamp,
            'cpu'        : [],
            'disk'       : [],
            'energy'     : False,
            'isolation'  : [],
            'total_calls': 0,
        }

        # 1. CPU scheduling via renice
        algorithm = decision.get('cpu_algorithm', 'RR')
        if len(proc_df) > 0:
            cpu_enforced = self.enforce_cpu_algorithm(
                algorithm, proc_df
            )
            results['cpu'] = cpu_enforced

        # 2. Disk I/O via ionice
        disk_pattern = decision.get('disk_pattern', 'mixed')
        if len(proc_df) > 0:
            disk_enforced = self.enforce_disk_pattern(
                disk_pattern, proc_df
            )
            results['disk'] = disk_enforced

        # 3. CPU frequency via cpufreq
        governor = decision.get('energy_governor', 'balanced')
        energy_ok = self.enforce_energy_governor(governor)
        results['energy'] = energy_ok

        # 4. Anomaly isolation
        anomalies = decision.get('anomaly_list', [])
        if anomalies:
            isolated = self.enforce_anomaly_isolation(
                anomalies
            )
            results['isolation'] = isolated

        # Count total kernel calls made
        results['total_calls'] = (
            len(results['cpu']) +
            len(results['disk']) +
            len(results['isolation']) +
            (1 if results['energy'] else 0)
        )

        # Log enforcement summary
        self.enforcement_log.append(results)
        if results['total_calls'] > 0:
            logger.info(
                f"Enforced: "
                f"CPU={len(results['cpu'])} renice | "
                f"Disk={len(results['disk'])} ionice | "
                f"Energy={'OK' if energy_ok else 'logged'} | "
                f"Isolated={len(results['isolation'])}"
            )

        return results

    def get_enforcement_summary(self):
        """Return summary of all enforcement actions taken."""
        if not self.enforcement_log:
            return {}

        total_cpu      = sum(
            len(r['cpu'])       for r in self.enforcement_log
        )
        total_disk     = sum(
            len(r['disk'])      for r in self.enforcement_log
        )
        total_isolated = sum(
            len(r['isolation']) for r in self.enforcement_log
        )
        total_calls    = sum(
            r['total_calls']    for r in self.enforcement_log
        )

        return {
            'total_cycles'     : len(self.enforcement_log),
            'total_kernel_calls': total_calls,
            'cpu_renice_calls' : total_cpu,
            'disk_ionice_calls': total_disk,
            'anomaly_isolated' : total_isolated,
            'capabilities'     : self.capabilities,
        }

    def save_log(self):
        """Save enforcement log to CSV for paper results."""
        if not self.enforcement_log:
            return
        rows = []
        for r in self.enforcement_log:
            rows.append({
                'timestamp'    : r['timestamp'],
                'cpu_calls'    : len(r['cpu']),
                'disk_calls'   : len(r['disk']),
                'energy_ok'    : r['energy'],
                'isolated'     : len(r['isolation']),
                'total_calls'  : r['total_calls'],
            })
        path = os.path.join(
            RESULT_DIR, 'enforcement_log.csv'
        )
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Enforcement log saved → {path}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import psutil

    print("\n" + "="*55)
    print("  AIMOS — Kernel Enforcement Layer Test")
    print("="*55 + "\n")

    enforcer = AIMOSKernelEnforcer()

    # Get real process data
    rows = []
    for proc in psutil.process_iter([
        'pid', 'name', 'cpu_percent',
        'nice', 'io_counters', 'memory_info'
    ]):
        try:
            info = proc.info
            rows.append({
                'pid'           : info['pid'],
                'name'          : info['name'],
                'cpu_percent'   : info['cpu_percent'] or 0,
                'nice'          : info['nice'] or 0,
                'io_read_bytes' : (
                    info['io_counters'].read_bytes
                    if info['io_counters'] else 0
                ),
                'io_write_bytes': (
                    info['io_counters'].write_bytes
                    if info['io_counters'] else 0
                ),
                'cpu_time_ns'   : 0,
                'wait_time_ns'  : 0,
            })
        except (psutil.NoSuchProcess,
                psutil.AccessDenied):
            continue

    proc_df = pd.DataFrame(rows)
    print(f"Processes loaded: {len(proc_df)}\n")

    # Test full decision enforcement
    test_decision = {
        'cpu_algorithm'  : 'PRIORITY',
        'energy_governor': 'balanced',
        'disk_pattern'   : 'read_heavy',
        'anomaly_list'   : [],
        'anomaly_count'  : 0,
    }

    print("--- Testing full enforcement cycle ---")
    results = enforcer.enforce_decision(
        test_decision, proc_df
    )

    print(f"\nEnforcement results:")
    print(f"  CPU renice calls  : {len(results['cpu'])}")
    print(f"  Disk ionice calls : {len(results['disk'])}")
    print(f"  Energy governor   : "
          f"{'applied' if results['energy'] else 'logged'}")
    print(f"  Anomaly isolated  : {len(results['isolation'])}")
    print(f"  Total kernel calls: {results['total_calls']}")

    if results['cpu']:
        print(f"\n  Processes reniced:")
        for r in results['cpu'][:5]:
            print(f"    {r['name']:<20} "
                  f"pid={r['pid']:<6} "
                  f"nice={r['nice']}")

    # Test all algorithms
    print("\n--- Testing all scheduling algorithms ---")
    algos = ['FCFS', 'SJF', 'RR', 'PRIORITY']
    for algo in algos:
        d = {'cpu_algorithm': algo,
             'energy_governor': 'balanced',
             'disk_pattern': 'mixed',
             'anomaly_list': []}
        r = enforcer.enforce_decision(d, proc_df)
        print(f"  {algo:<10}: {r['total_calls']} kernel calls")

    # Current governor
    gov = enforcer.read_current_governor()
    print(f"\n  Current CPU governor: {gov}")

    # Summary
    summary = enforcer.get_enforcement_summary()
    print(f"\n--- Enforcement summary ---")
    for k, v in summary.items():
        if k != 'capabilities':
            print(f"  {k:<25}: {v}")

    print(f"\n  Capabilities:")
    for cap, avail in summary['capabilities'].items():
        status = "AVAILABLE" if avail else "WSL2-limited"
        print(f"    {cap:<12}: {status}")

    enforcer.save_log()

    print(f"\n{'='*55}")
    print(f"  Fix 2 COMPLETE — Layer 4 gap closed")
    print(f"  AI decisions now enforced via kernel interfaces")
    print(f"{'='*55}\n")
