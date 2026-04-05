"""
AIMOS — Module 1: CPU Scheduler RL Environment
Custom OpenAI Gym environment for CPU scheduling decisions.
The agent observes real process metrics and selects the best
scheduling algorithm dynamically.

Algorithms the agent can choose:
  0 = FCFS     (First Come First Served)
  1 = SJF      (Shortest Job First)
  2 = RR       (Round Robin)
  3 = PRIORITY (Priority-based)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

from utils.config import RAW_METRICS_CSV, CPU_ALGORITHMS


class CPUSchedulerEnv(gym.Env):
    """
    Custom Gym environment for CPU scheduling.

    State space (what the agent observes):
        - avg_cpu_percent     : average CPU usage across processes
        - avg_wait_time_norm  : normalized average wait time
        - avg_burst_ratio     : how bursty the workload is
        - interactive_ratio   : fraction of short/interactive processes
        - cpu_load_norm       : system load average (normalized)
        - queue_length_norm   : number of runnable processes (normalized)
        - io_bound_ratio      : fraction of I/O heavy processes

    Action space (what the agent decides):
        0 = FCFS
        1 = SJF
        2 = RR
        3 = PRIORITY

    Reward:
        Positive when average waiting time decreases.
        Negative when waiting time increases or throughput drops.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data_path=RAW_METRICS_CSV, window=20):
        super().__init__()

        self.data_path   = data_path
        self.window      = window       # how many snapshots per step
        self.current_idx = 0
        self.data        = None
        self.prev_avg_wait = 0.0

        # ── Action space: 4 scheduling algorithms ─────────────────────────
        self.action_space = spaces.Discrete(len(CPU_ALGORITHMS))
        self.algorithm_names = CPU_ALGORITHMS

        # ── Observation space: 7 normalized features [0, 1] ───────────────
        self.observation_space = spaces.Box(
            low   = np.zeros(7, dtype=np.float32),
            high  = np.ones(7,  dtype=np.float32),
            dtype = np.float32
        )

        # ── Load and preprocess data ───────────────────────────────────────
        self._load_data()

        # ── Tracking for evaluation ───────────────────────────────────────
        self.episode_rewards    = []
        self.episode_actions    = []
        self.episode_wait_times = []

    def _load_data(self):
        """Load collected OS metrics and prepare features."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"No data at {self.data_path}. "
                f"Run run_collector.py first."
            )

        df = pd.read_csv(self.data_path)
        proc_df = df[df['type'] == 'process'].copy()

        # Fill missing values
        proc_df.fillna(0, inplace=True)

        # Group by timestamp — each group = one scheduling snapshot
        proc_df['timestamp'] = pd.to_datetime(proc_df['timestamp'])
        self.snapshots = []

        for ts, group in proc_df.groupby('timestamp'):
            if len(group) < 2:
                continue

            snap = self._extract_features(group)
            if snap is not None:
                self.snapshots.append(snap)

        if len(self.snapshots) < 10:
            # If not enough real data, generate synthetic
            self.snapshots = self._generate_synthetic_data(500)
            print("[CPUSchedulerEnv] Using synthetic data "
                  "(collect more real data for better training)")
        else:
            print(f"[CPUSchedulerEnv] Loaded {len(self.snapshots)} "
                  f"real snapshots from {self.data_path}")

        self.snapshots = np.array(self.snapshots, dtype=np.float32)

    def _extract_features(self, group):
        """Extract 7 normalized features from a process group."""
        try:
            n = len(group)
            if n == 0:
                return None

            cpu_vals  = group['cpu_percent'].values
            wait_vals = group['wait_time_ns'].values
            nice_vals = group['nice'].values
            io_read   = group['io_read_bytes'].values
            io_write  = group['io_write_bytes'].values

            # Feature 1: average CPU usage (normalized 0-100 → 0-1)
            avg_cpu = np.clip(np.mean(cpu_vals) / 100.0, 0, 1)

            # Feature 2: average wait time normalized (ns → 0-1)
            max_wait = 1e10
            avg_wait = np.clip(np.mean(wait_vals) / max_wait, 0, 1)

            # Feature 3: burst ratio
            # high variance in CPU = bursty workload
            burst = np.clip(np.std(cpu_vals) / 50.0, 0, 1)

            # Feature 4: interactive ratio
            # processes with low CPU and low wait = interactive
            interactive = np.mean(cpu_vals < 5.0)

            # Feature 5: system load (from wait time distribution)
            load = np.clip(np.percentile(wait_vals, 75) / max_wait, 0, 1)

            # Feature 6: queue length (normalized, assume max 200 procs)
            queue_len = np.clip(n / 200.0, 0, 1)

            # Feature 7: I/O bound ratio
            total_io  = io_read + io_write
            io_bound  = np.mean(total_io > np.median(total_io + 1))

            return [avg_cpu, avg_wait, burst,
                    interactive, load, queue_len, io_bound]

        except Exception:
            return None

    def _generate_synthetic_data(self, n=500):
        """
        Generate synthetic but realistic workload scenarios.
        Used when real data is insufficient.
        4 workload types: idle, interactive, cpu-bound, mixed
        """
        data = []
        np.random.seed(42)

        for _ in range(n):
            workload = np.random.choice(
                ['idle', 'interactive', 'cpu_bound', 'mixed'],
                p=[0.2, 0.3, 0.3, 0.2]
            )

            if workload == 'idle':
                snap = [
                    np.random.uniform(0.0, 0.05),   # low cpu
                    np.random.uniform(0.0, 0.05),   # low wait
                    np.random.uniform(0.0, 0.05),   # low burst
                    np.random.uniform(0.7, 1.0),    # high interactive
                    np.random.uniform(0.0, 0.05),   # low load
                    np.random.uniform(0.0, 0.1),    # few procs
                    np.random.uniform(0.0, 0.1),    # low io
                ]
            elif workload == 'interactive':
                snap = [
                    np.random.uniform(0.1, 0.4),
                    np.random.uniform(0.05, 0.2),
                    np.random.uniform(0.1, 0.4),
                    np.random.uniform(0.5, 0.9),
                    np.random.uniform(0.1, 0.3),
                    np.random.uniform(0.2, 0.5),
                    np.random.uniform(0.3, 0.7),
                ]
            elif workload == 'cpu_bound':
                snap = [
                    np.random.uniform(0.6, 1.0),
                    np.random.uniform(0.3, 0.8),
                    np.random.uniform(0.4, 0.9),
                    np.random.uniform(0.0, 0.2),
                    np.random.uniform(0.5, 0.9),
                    np.random.uniform(0.4, 0.8),
                    np.random.uniform(0.0, 0.2),
                ]
            else:  # mixed
                snap = [
                    np.random.uniform(0.2, 0.7),
                    np.random.uniform(0.1, 0.5),
                    np.random.uniform(0.2, 0.6),
                    np.random.uniform(0.2, 0.6),
                    np.random.uniform(0.2, 0.6),
                    np.random.uniform(0.3, 0.7),
                    np.random.uniform(0.2, 0.5),
                ]

            data.append([float(np.clip(x, 0, 1)) for x in snap])

        return data

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset environment to start of dataset."""
        super().reset(seed=seed)
        self.current_idx   = 0
        self.prev_avg_wait = 0.0
        self.episode_rewards    = []
        self.episode_actions    = []
        self.episode_wait_times = []

        obs = self.snapshots[self.current_idx]
        return obs, {}

    def step(self, action):
        """
        Apply scheduling decision and return outcome.

        The reward function:
          - Compare current wait time to previous wait time
          - Good action (wait time drops) → positive reward
          - Bad action (wait time rises) → negative reward
          - Bonus for choosing right algorithm for workload type
        """
        obs     = self.snapshots[self.current_idx]
        reward  = self._compute_reward(action, obs)
        done    = False

        # Track for evaluation
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_wait_times.append(float(obs[1]))

        # Advance to next snapshot
        self.current_idx += 1
        if self.current_idx >= len(self.snapshots) - 1:
            done             = True
            self.current_idx = 0

        next_obs = self.snapshots[self.current_idx]
        self.prev_avg_wait = float(obs[1])

        return next_obs, reward, done, False, {
            'algorithm'   : self.algorithm_names[action],
            'wait_time'   : float(obs[1]),
            'cpu_load'    : float(obs[0]),
            'step'        : self.current_idx,
        }

    def _compute_reward(self, action, obs):
        """
        Reward logic — this encodes OS scheduling knowledge into the agent.

        Key insight:
          FCFS     → good for batch, low-load workloads
          SJF      → good for minimizing average wait time
          RR       → good for interactive, mixed workloads
          PRIORITY → good for high-load, cpu-bound workloads
        """
        avg_cpu     = obs[0]
        avg_wait    = obs[1]
        burst       = obs[2]
        interactive = obs[3]
        load        = obs[4]
        queue_len   = obs[5]
        io_bound    = obs[6]

        # Base reward: improvement in wait time
        wait_delta = self.prev_avg_wait - avg_wait
        base_reward = wait_delta * 10.0

        # Algorithm-workload alignment bonus
        bonus = 0.0

        if action == 0:  # FCFS
            # Good when load is low and workload is simple
            if load < 0.2 and burst < 0.2:
                bonus = 0.5
            else:
                bonus = -0.3    # FCFS hurts under high load

        elif action == 1:  # SJF
            # Good when processes have predictable burst times
            if burst < 0.4 and interactive > 0.4:
                bonus = 0.6
            elif avg_cpu > 0.7:
                bonus = -0.2    # SJF struggles with cpu-bound

        elif action == 2:  # Round Robin
            # Good for interactive and mixed workloads
            if interactive > 0.5 or (0.2 < avg_cpu < 0.7):
                bonus = 0.7
            elif avg_cpu < 0.1:
                bonus = -0.1    # RR overhead not worth it when idle

        elif action == 3:  # PRIORITY
            # Good for high-load cpu-bound workloads
            if avg_cpu > 0.6 or load > 0.6:
                bonus = 0.8
            elif interactive > 0.7:
                bonus = -0.2    # Priority can starve interactive procs

        # Penalty for keeping long wait times
        wait_penalty = -avg_wait * 2.0

        total_reward = base_reward + bonus + wait_penalty
        return float(np.clip(total_reward, -2.0, 2.0))

    def render(self, mode='human'):
        obs = self.snapshots[self.current_idx]
        print(f"\n[AIMOS Scheduler] Step {self.current_idx}")
        print(f"  CPU load     : {obs[0]:.3f}")
        print(f"  Avg wait     : {obs[1]:.3f}")
        print(f"  Burst ratio  : {obs[2]:.3f}")
        print(f"  Interactive  : {obs[3]:.3f}")
        print(f"  Queue length : {obs[5]:.3f}")

    def get_episode_summary(self):
        if not self.episode_rewards:
            return {}
        actions = self.episode_actions
        return {
            'total_reward'      : sum(self.episode_rewards),
            'avg_reward'        : np.mean(self.episode_rewards),
            'avg_wait_time'     : np.mean(self.episode_wait_times),
            'fcfs_chosen'       : actions.count(0),
            'sjf_chosen'        : actions.count(1),
            'rr_chosen'         : actions.count(2),
            'priority_chosen'   : actions.count(3),
            'steps'             : len(actions),
        }
