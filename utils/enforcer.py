"""
AIMOS — Kernel Enforcement Layer
Writes AI decisions back to the real Linux kernel
via cgroups, renice, and cpufreq interfaces.
This is what makes AIMOS a genuine feedback loop.
"""

import os
import subprocess
import logging
import psutil

logger = logging.getLogger(__name__)

class AIMOSEnforcer:

    def __init__(self):
        self.cgroup_path = "/sys/fs/cgroup"
        self.aimos_cgroup = os.path.join(self.cgroup_path, "aimos")
        self._setup_cgroup()

    def _setup_cgroup(self):
        """Create AIMOS cgroup if it doesn't exist."""
        try:
            os.makedirs(self.aimos_cgroup, exist_ok=True)
            logger.info(f"AIMOS cgroup ready at {self.aimos_cgroup}")
        except PermissionError:
            logger.warning("No root access — enforcement limited to renice")
        except Exception as e:
            logger.warning(f"cgroup setup: {e}")

    # ── CPU enforcement ───────────────────────────────────────────────────────
    def set_process_priority(self, pid: int, priority: int):
        """
        Change real process priority via renice.
        priority: -20 (highest) to 19 (lowest)
        Works without root for processes you own.
        """
        try:
            priority = max(-20, min(19, priority))
            proc = psutil.Process(pid)
            proc.nice(priority)
            logger.info(f"Set PID {pid} priority → {priority}")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Priority set failed PID {pid}: {e}")
            return False

    def set_cpu_affinity(self, pid: int, cores: list):
        """Pin a process to specific CPU cores."""
        try:
            proc = psutil.Process(pid)
            proc.cpu_affinity(cores)
            logger.info(f"Set PID {pid} CPU affinity → cores {cores}")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Affinity set failed PID {pid}: {e}")
            return False

    def set_cpu_quota(self, pid: int, quota_percent: int):
        """
        Limit process CPU usage via cgroup cpu.max
        quota_percent: 0-100 (% of one CPU)
        Requires root.
        """
        try:
            period = 100000
            quota  = int(period * quota_percent / 100)
            proc_cgroup = os.path.join(
                self.aimos_cgroup, f"proc_{pid}")
            os.makedirs(proc_cgroup, exist_ok=True)

            # Add process to cgroup
            with open(os.path.join(proc_cgroup, "cgroup.procs"), 'w') as f:
                f.write(str(pid))

            # Set CPU limit
            with open(os.path.join(proc_cgroup, "cpu.max"), 'w') as f:
                f.write(f"{quota} {period}")

            logger.info(f"Set PID {pid} CPU quota → {quota_percent}%")
            return True
        except Exception as e:
            logger.warning(f"CPU quota failed PID {pid}: {e}")
            return False

    # ── Memory enforcement ────────────────────────────────────────────────────
    def set_memory_limit(self, pid: int, limit_mb: int):
        """
        Limit process memory via cgroup memory.max
        Requires root.
        """
        try:
            proc_cgroup = os.path.join(
                self.aimos_cgroup, f"proc_{pid}")
            os.makedirs(proc_cgroup, exist_ok=True)

            with open(os.path.join(proc_cgroup, "cgroup.procs"), 'w') as f:
                f.write(str(pid))

            limit_bytes = limit_mb * 1024 * 1024
            with open(os.path.join(proc_cgroup, "memory.max"), 'w') as f:
                f.write(str(limit_bytes))

            logger.info(f"Set PID {pid} memory limit → {limit_mb}MB")
            return True
        except Exception as e:
            logger.warning(f"Memory limit failed PID {pid}: {e}")
            return False

    # ── I/O enforcement ───────────────────────────────────────────────────────
    def set_io_priority(self, pid: int, ioclass: int = 2, level: int = 4):
        """
        Set I/O scheduling priority via ionice.
        ioclass: 1=realtime, 2=best-effort, 3=idle
        level: 0(highest)-7(lowest)
        """
        try:
            result = subprocess.run(
                ["ionice", "-c", str(ioclass),
                 "-n", str(level), "-p", str(pid)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(
                    f"Set PID {pid} I/O priority → "
                    f"class={ioclass} level={level}")
                return True
            else:
                logger.warning(f"ionice failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("ionice not found — install util-linux")
            return False

    # ── Energy enforcement ────────────────────────────────────────────────────
    def set_cpu_governor(self, governor: str = "schedutil"):
        """
        Set CPU frequency governor.
        Options: performance, powersave, schedutil, ondemand
        Requires root. Not available in WSL2.
        """
        try:
            cpu_count = psutil.cpu_count()
            for i in range(cpu_count):
                gov_path = (f"/sys/devices/system/cpu/cpu{i}"
                            f"/cpufreq/scaling_governor")
                if os.path.exists(gov_path):
                    with open(gov_path, 'w') as f:
                        f.write(governor)
            logger.info(f"Set CPU governor → {governor}")
            return True
        except PermissionError:
            logger.warning("cpufreq control needs root / not in WSL2")
            return False
        except Exception as e:
            logger.warning(f"Governor set failed: {e}")
            return False

    def set_cpu_frequency(self, freq_khz: int):
        """Set CPU frequency directly (needs root + userspace governor)."""
        try:
            cpu_count = psutil.cpu_count()
            for i in range(cpu_count):
                freq_path = (f"/sys/devices/system/cpu/cpu{i}"
                             f"/cpufreq/scaling_setspeed")
                if os.path.exists(freq_path):
                    with open(freq_path, 'w') as f:
                        f.write(str(freq_khz))
            logger.info(f"Set CPU frequency → {freq_khz} kHz")
            return True
        except Exception as e:
            logger.warning(f"Frequency set failed: {e}")
            return False

    # ── Anomaly response ──────────────────────────────────────────────────────
    def throttle_suspicious_process(self, pid: int, score: float):
        """
        Respond to anomaly detection:
        score > 0.9 → kill process (with safety check)
        score > 0.7 → throttle to 20% CPU
        score > 0.5 → lower priority to 10
        """
        try:
            proc = psutil.Process(pid)
            name = proc.name()

            # Safety: never touch system processes
            protected = {'systemd','init','sshd','bash',
                        'python','python3','kernel'}
            if name in protected:
                logger.info(
                    f"Skipping protected process: {name} (PID {pid})")
                return False

            if score > 0.9:
                logger.warning(
                    f"ANOMALY KILL: PID {pid} ({name}) score={score:.2f}")
                proc.terminate()
                return True
            elif score > 0.7:
                self.set_cpu_quota(pid, 20)
                logger.warning(
                    f"ANOMALY THROTTLE: PID {pid} ({name}) "
                    f"→ 20% CPU score={score:.2f}")
                return True
            elif score > 0.5:
                self.set_process_priority(pid, 10)
                logger.info(
                    f"ANOMALY DEPRIORITIZE: PID {pid} ({name}) "
                    f"score={score:.2f}")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return False


def test_enforcer():
    """Quick test — verify enforcement works on your system."""
    import os
    enforcer = AIMOSEnforcer()
    pid = os.getpid()

    print("\n=== AIMOS Enforcer Test ===\n")

    r1 = enforcer.set_process_priority(pid, 5)
    print(f"renice self to 5        : {'OK' if r1 else 'FAILED'}")

    r2 = enforcer.set_cpu_affinity(pid, [0, 1])
    print(f"pin to cores 0,1        : {'OK' if r2 else 'FAILED'}")

    r3 = enforcer.set_io_priority(pid, ioclass=2, level=4)
    print(f"ionice best-effort      : {'OK' if r3 else 'FAILED'}")

    r4 = enforcer.set_cpu_governor("schedutil")
    print(f"cpufreq governor        : {'OK' if r4 else 'N/A (WSL2)'}")

    print("\nEnforcer test complete.")
    print("Note: cgroup quota/memory limits need root (sudo).")

if __name__ == '__main__':
    test_enforcer()
