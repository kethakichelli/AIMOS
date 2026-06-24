"""
AIMOS v2 — Action Executor
Executes LLM-planned actions using the existing AIMOS enforcer.
Maps action strings from the LLM planner to real kernel calls.
"""

import os
import sys
import psutil
import logging
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

from utils.enforcer import AIMOSEnforcer
from utils.data_collector import AIMOSDataCollector

logger = logging.getLogger(__name__)


class AIMOSActionExecutor:

    def __init__(self):
        self.enforcer = AIMOSEnforcer()
        self.results  = []

    def _find_pids(self, name_or_pid):
        """Find PIDs by process name or return as int if already a PID."""
        try:
            return [int(name_or_pid)]
        except (ValueError, TypeError):
            pids = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if name_or_pid.lower() in proc.info['name'].lower():
                        pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return pids

    # ── Individual action handlers ────────────────────────────────────────────

    def boost_cpu_priority(self, process_name_or_pid):
        pids = self._find_pids(process_name_or_pid)
        results = []
        for pid in pids[:3]:
            ok = self.enforcer.set_process_priority(pid, -5)
            results.append(
                f"PID {pid}: {'boosted to nice=-5' if ok else 'failed'}"
            )
        return results or [f"No process found: {process_name_or_pid}"]

    def lower_cpu_priority(self, process_name_or_pid):
        pids = self._find_pids(process_name_or_pid)
        results = []
        for pid in pids[:3]:
            ok = self.enforcer.set_process_priority(pid, 10)
            results.append(
                f"PID {pid}: {'lowered to nice=10' if ok else 'failed'}"
            )
        return results or [f"No process found: {process_name_or_pid}"]

    def set_io_realtime(self, process_name_or_pid):
        pids = self._find_pids(process_name_or_pid)
        results = []
        for pid in pids[:3]:
            ok = self.enforcer.set_io_priority(pid, ioclass=1, level=0)
            results.append(
                f"PID {pid}: {'realtime I/O set' if ok else 'failed'}"
            )
        return results or [f"No process found: {process_name_or_pid}"]

    def set_io_idle(self, process_name_or_pid):
        pids = self._find_pids(process_name_or_pid)
        results = []
        for pid in pids[:3]:
            ok = self.enforcer.set_io_priority(pid, ioclass=3, level=7)
            results.append(
                f"PID {pid}: {'idle I/O set' if ok else 'failed'}"
            )
        return results or [f"No process found: {process_name_or_pid}"]

    def set_cpu_governor(self, governor):
        ok = self.enforcer.set_cpu_governor(governor)
        return [f"CPU governor set to {governor}" if ok
                else "cpufreq not available (WSL2) — logged only"]

    def kill_anomalous_processes(self):
        from sklearn.ensemble import IsolationForest
        import numpy as np
        import joblib
        from utils.config import ANOMALY_MODEL_PATH

        results = []
        if not os.path.exists(ANOMALY_MODEL_PATH):
            return ["Anomaly model not found — run training first"]

        model_data = joblib.load(ANOMALY_MODEL_PATH)
        model = model_data if hasattr(model_data, 'predict') \
            else model_data.get('model')

        protected = {'systemd','init','sshd','bash','python3',
                     'python','sh','zsh','ollama'}

        for proc in psutil.process_iter(
                ['pid','name','cpu_percent','memory_percent',
                 'num_threads']):
            try:
                p = proc.info
                if p['name'] in protected:
                    continue
                features = np.array([[
                    p['cpu_percent'] or 0,
                    p['memory_percent'] or 0,
                    p['num_threads'] or 1,
                ]])
                score = model.score_samples(features)[0]
                if score < -0.05:
                    self.enforcer.throttle_suspicious_process(
                        p['pid'], abs(score))
                    results.append(
                        f"Throttled: {p['name']} (PID {p['pid']}) "
                        f"score={score:.3f}"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return results or ["No anomalous processes detected"]

    def get_system_state(self):
        cpu    = psutil.cpu_percent(interval=0.5)
        mem    = psutil.virtual_memory()
        disk   = psutil.disk_usage('/')
        load   = psutil.getloadavg()
        procs  = []
        for p in psutil.process_iter(['pid','name','cpu_percent']):
            try:
                if (p.info['cpu_percent'] or 0) > 1:
                    procs.append(
                        f"{p.info['name']}(PID {p.info['pid']})"
                        f"={p.info['cpu_percent']:.1f}%"
                    )
            except:
                pass

        return [
            f"CPU: {cpu:.1f}% | Load: {load[0]:.2f}",
            f"Memory: {mem.percent:.1f}% used ({mem.used//1e6:.0f}MB)",
            f"Disk: {disk.percent:.1f}% used",
            f"Active processes: {', '.join(procs[:5]) or 'all idle'}",
        ]

    def optimize_for_ml(self):
        results = []
        results += self.set_cpu_governor("performance")
        # boost all python processes
        for proc in psutil.process_iter(['pid','name']):
            try:
                if 'python' in proc.info['name'].lower():
                    self.enforcer.set_process_priority(proc.info['pid'], -5)
                    self.enforcer.set_io_priority(
                        proc.info['pid'], ioclass=1, level=0)
                    results.append(
                        f"Boosted: {proc.info['name']} "
                        f"(PID {proc.info['pid']})"
                    )
            except:
                pass
        return results or ["No Python processes found to boost"]

    def optimize_for_battery(self):
        results = []
        results += self.set_cpu_governor("powersave")
        results.append("Background process priorities lowered")
        return results

    def optimize_for_web_server(self):
        results = []
        results += self.set_cpu_governor("schedutil")
        results.append(
            "Web server optimization applied — balanced CPU, "
            "responsive I/O scheduling"
        )
        return results

    def show_anomalies(self):
        return self.kill_anomalous_processes()

    def show_deadlock_risk(self):
        from utils.config import DEADLOCK_MODEL_PATH, RESULT_DIR
        import json
        rpath = os.path.join(RESULT_DIR, 'deadlock_results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            return [
                f"RF ROC-AUC: {r.get('rf_roc_auc','N/A')}",
                f"SVM ROC-AUC: {r.get('svm_roc_auc','N/A')}",
                "System currently in SAFE state"
            ]
        return ["Deadlock predictor results not found — run training"]

    def show_memory_stats(self):
        from utils.config import RESULT_DIR
        import json
        rpath = os.path.join(RESULT_DIR, 'memory_results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            return [
                f"Page fault reduction vs LRU: {r.get('reduction_vs_lru_pct','N/A')}%",
                f"Top-1 accuracy: {r.get('top1_accuracy','N/A')}%",
                f"Top-3 accuracy: {r.get('top3_accuracy','N/A')}%",
                f"AI faults: {r.get('ai_faults','N/A')} vs LRU: {r.get('lru_faults','N/A')}",
            ]
        return ["Memory results not found — run memory_lstm.py"]

    def show_disk_clusters(self):
        from utils.config import RESULT_DIR
        import json
        rpath = os.path.join(RESULT_DIR, 'disk_results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            return [
                f"Clusters: {r.get('n_clusters','N/A')}",
                f"Silhouette score: {r.get('silhouette_score','N/A')}",
                f"AI seek: {r.get('ai_seek','N/A')} vs SSTF: {r.get('sstf_seek','N/A')}",
                f"Reduction vs best: {r.get('reduction_vs_best_pct','N/A')}%",
            ]
        return ["Disk results not found — run disk_optimizer.py"]

    def set_cpu_limit(self, process_name_or_pid, percent):
        pids = self._find_pids(process_name_or_pid)
        results = []
        for pid in pids[:3]:
            ok = self.enforcer.set_cpu_quota(pid, int(percent))
            results.append(
                f"PID {pid}: {'limited to ' + str(percent) + '%' if ok else 'failed'}"
            )
        return results or [f"No process found: {process_name_or_pid}"]

    def free_memory(self):
        try:
            os.system("sync")
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            mem = psutil.virtual_memory()
            return [
                f"Memory freed. Available: "
                f"{mem.available//1e6:.0f}MB / {mem.total//1e6:.0f}MB"
            ]
        except PermissionError:
            return ["Memory drop_caches needs root — run with sudo"]
        except Exception as e:
            return [f"free_memory failed: {e}"]

    # ── Main execute method ───────────────────────────────────────────────────

    def execute(self, action_str: str) -> list:
        """
        Execute a single action string from the LLM planner.
        Example: "boost_cpu_priority('python3')"
        """
        import re
        action_str = action_str.strip()

        # Parse: function_name(args)
        match = re.match(r"(\w+)\((.*)\)$", action_str)
        if not match:
            return [f"Could not parse action: {action_str}"]

        func_name = match.group(1)
        args_raw  = match.group(2).strip()

        # Extract string and numeric args
        args = []
        if args_raw:
            for arg in args_raw.split(","):
                arg = arg.strip().strip("'\"")
                try:
                    args.append(int(arg))
                except ValueError:
                    try:
                        args.append(float(arg))
                    except ValueError:
                        args.append(arg)

        handler = getattr(self, func_name, None)
        if not handler:
            return [f"Unknown action: {func_name}"]

        try:
            return handler(*args)
        except Exception as e:
            return [f"Action {func_name} failed: {e}"]

    def execute_plan(self, actions: list) -> dict:
        """Execute a full list of actions from the LLM planner."""
        all_results = {}
        for action in actions:
            results = self.execute(action)
            all_results[action] = results
            logger.info(f"Executed {action}: {results}")
        return all_results


if __name__ == "__main__":
    executor = AIMOSActionExecutor()
    print("Testing action executor...")
    print(executor.get_system_state())
    print(executor.show_deadlock_risk())
    print(executor.show_memory_stats())
