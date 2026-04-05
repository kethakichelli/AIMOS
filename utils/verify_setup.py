"""
AIMOS — Environment verification script.
Run this any time to confirm your setup is intact.
"""

import sys
import os
import subprocess

print()
print("=" * 50)
print("  AIMOS — Environment Check")
print("=" * 50)

# ── Python version ────────────────────────────────
print(f"\n[Python]")
print(f"  Version   : {sys.version.split()[0]}")
print(f"  Executable: {sys.executable}")

# ── Core libraries ────────────────────────────────
print(f"\n[Libraries]")
libs = [
    "psutil", "pandas", "numpy", "sklearn",
    "torch", "gymnasium", "stable_baselines3",
    "streamlit", "plotly", "scipy", "seaborn"
]
all_ok = True
for lib in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, '__version__', 'ok')
        print(f"  {lib:<22}: {ver}")
    except ImportError:
        print(f"  {lib:<22}: MISSING ← run pip install {lib}")
        all_ok = False

# ── System metrics ────────────────────────────────
print(f"\n[System]")
import psutil
print(f"  CPU cores     : {psutil.cpu_count()}")
print(f"  RAM (GB)      : {round(psutil.virtual_memory().total / 1e9, 2)}")
print(f"  Disk (GB)     : {round(psutil.disk_usage('/').total / 1e9, 2)}")

# ── /proc filesystem ──────────────────────────────
print(f"\n[/proc interface]")
proc_files = [
    '/proc/stat',
    '/proc/meminfo',
    '/proc/diskstats',
    '/proc/1/schedstat',
]
for f in proc_files:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"  {f:<30}: {status}")

# ── bpftrace / eBPF ───────────────────────────────
print(f"\n[eBPF / bpftrace]")
try:
    r = subprocess.run(
        ['bpftrace', '--version'],
        capture_output=True, text=True, timeout=5
    )
    print(f"  bpftrace      : {r.stdout.strip()}")
    r2 = subprocess.run(
        ['sudo', 'bpftrace', '-e',
         'BEGIN { printf("probe OK\\n"); exit(); }'],
        capture_output=True, text=True, timeout=10
    )
    probe = "OK" if "probe OK" in r2.stdout else "needs sudo"
    print(f"  eBPF probe    : {probe}")
except FileNotFoundError:
    print("  bpftrace      : NOT INSTALLED")
    print("  Run: sudo apt install -y bpftrace")

# ── AIMOS project structure ───────────────────────
print(f"\n[AIMOS project folders]")
base = os.path.expanduser("~/AIMOS")
folders = ['data','models','modules','dashboard',
           'utils','logs','results','paper']
for folder in folders:
    path   = os.path.join(base, folder)
    status = "OK" if os.path.isdir(path) else "MISSING"
    print(f"  ~/AIMOS/{folder:<12}: {status}")

# ── Final verdict ─────────────────────────────────
print()
if all_ok:
    print("  ALL SYSTEMS GO — ready to build AIMOS")
else:
    print("  Some libraries missing — re-run pip install")
print("=" * 50)
print()
