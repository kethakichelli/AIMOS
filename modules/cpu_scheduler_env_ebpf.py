"""
AIMOS — Module 1 v2: CPU Scheduler RL Environment
Retrained specifically on eBPF kernel observations.
Closes the Layer 3 gap — agent now trained on the exact
same feature vector it receives at inference time.
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from utils.config import (
    CPU_ALGORITHMS, LOG_DIR, RAW_METRICS_CSV
)


class CPUSchedulerEnvEBPF(gym.Env):
    """
    CPU Scheduler RL Environment trained on eBPF observations.

    Observation vector (7 features) — matches exactly what
    AIMOSeBPFCollector.get_rl_observation() returns at runtime:
        0: cpu_bound_ratio       — fraction of CPU-bound processes
        1: avg_wait_norm         — normalised average wait time
        2: preempt_ratio         — how often processes are preempted
        3: interactive_ratio     — fraction of interactive processes
        4: load_norm             — system load normalised
        5: queue_length_norm     — number of active processes
        6: migration_ratio       — CPU core migration rate

    Actions:
        0: FCFS
        1: SJF
        2: RR
        3: PRIORITY
    """

    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(len(CPU_ALGORITHMS))
        self.observation_space = spaces.Box(
            low   = np.zeros(7, dtype=np.float32),
            high  = np.ones(7,  dtype=np.float32),
            dtype = np.float32
        )
        self.algorithm_names = CPU_ALGORITHMS
        self.current_idx     = 0
        self.prev_wait       = 0.0
        self.episode_actions = []
        self.episode_rewards = []

        self.snapshots = self._load_data()
        print(f"[CPUSchedulerEnvEBPF] "
              f"Loaded {len(self.snapshots)} snapshots")

    def _load_data(self):
        """
        Load training data in priority order:
          1. eBPF feature log (best — real kernel data)
          2. /proc raw metrics (fallback)
          3. Synthetic workloads (last resort)
        """
        snapshots = []

        # Priority 1: eBPF feature log
        ebpf_path = os.path.join(
            LOG_DIR, 'ebpf_sched_features.csv'
        )
        if os.path.exists(ebpf_path):
            try:
                df = pd.read_csv(ebpf_path)
                snapshots = self._features_from_ebpf(df)
                if len(snapshots) >= 10:
                    print(f"[CPUSchedulerEnvEBPF] "
                          f"Using eBPF kernel data "
                          f"({len(snapshots)} snapshots)")
                    return np.array(snapshots, dtype=np.float32)
            except Exception as e:
                print(f"[CPUSchedulerEnvEBPF] "
                      f"eBPF load failed: {e}")

        # Priority 2: /proc raw metrics
        if os.path.exists(RAW_METRICS_CSV):
            try:
                df = pd.read_csv(RAW_METRICS_CSV)
                snapshots = self._features_from_proc(df)
                if len(snapshots) >= 10:
                    print(f"[CPUSchedulerEnvEBPF] "
                          f"Using /proc data "
                          f"({len(snapshots)} snapshots)")
                    return np.array(snapshots, dtype=np.float32)
            except Exception as e:
                print(f"[CPUSchedulerEnvEBPF] "
                      f"/proc load failed: {e}")

        # Priority 3: Synthetic
        print("[CPUSchedulerEnvEBPF] Using synthetic data")
        return np.array(
            self._generate_synthetic(600), dtype=np.float32
        )

    def _features_from_ebpf(self, df):
        """
        Convert eBPF feature log into 7-element observation vectors.
        Column mapping from ebpf_collector._take_feature_snapshot()
        """
        required = [
            'cpu_bound_ratio', 'avg_wait_ns',
            'avg_preempt_ratio', 'interactive_ratio',
            'active_procs', 'avg_affinity'
        ]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0

        snapshots = []
        for _, row in df.iterrows():
            obs = [
                float(np.clip(
                    row.get('cpu_bound_ratio', 0), 0, 1)),
                float(np.clip(
                    row.get('avg_wait_ns', 0) / 1e9, 0, 1)),
                float(np.clip(
                    row.get('avg_preempt_ratio', 0), 0, 1)),
                float(np.clip(
                    row.get('interactive_ratio', 0), 0, 1)),
                float(np.clip(
                    row.get('avg_wait_ns', 0) / 1e9, 0, 1)),
                float(np.clip(
                    row.get('active_procs', 0) / 200.0, 0, 1)),
                float(np.clip(
                    1.0 - row.get('avg_affinity', 1.0), 0, 1)),
            ]
            snapshots.append(obs)
        return snapshots

    def _features_from_proc(self, df):
        """Convert /proc metrics into eBPF-compatible observation."""
        proc_df = df[df['type'] == 'process'].copy()
        num_cols = proc_df.select_dtypes(
            include='number').columns
        proc_df[num_cols] = proc_df[num_cols].fillna(0)

        snapshots = []
        for ts, group in proc_df.groupby('timestamp'):
            if len(group) < 2:
                continue
            try:
                cpu   = group['cpu_percent'].values
                wait  = group['wait_time_ns'].values
                io_r  = group['io_read_bytes'].values
                io_w  = group['io_write_bytes'].values

                obs = [
                    float(np.clip(
                        np.mean(cpu > 50) , 0, 1)),
                    float(np.clip(
                        np.mean(wait) / 1e10, 0, 1)),
                    float(np.clip(
                        np.std(cpu) / 50.0, 0, 1)),
                    float(np.mean(cpu < 5.0)),
                    float(np.clip(
                        np.percentile(wait, 75) / 1e10,
                        0, 1)),
                    float(np.clip(
                        len(group) / 200.0, 0, 1)),
                    float(np.clip(
                        np.mean(
                            (io_r + io_w) > np.median(
                                io_r + io_w + 1
                            )
                        ), 0, 1)),
                ]
                snapshots.append(obs)
            except Exception:
                continue
        return snapshots

    def _generate_synthetic(self, n=600):
        """
        Rich synthetic workload generator.
        Covers 5 workload profiles to ensure agent sees
        all scheduling scenarios during training.
        """
        np.random.seed(42)
        data = []

        profiles = {
            'idle'       : (0.15, [
                (0.0,0.05),(0.0,0.03),(0.0,0.05),
                (0.8,1.0),(0.0,0.03),(0.0,0.08),(0.0,0.05)
            ]),
            'interactive': (0.25, [
                (0.05,0.25),(0.02,0.15),(0.05,0.25),
                (0.6,0.95),(0.02,0.15),(0.1,0.4),(0.1,0.3)
            ]),
            'cpu_bound'  : (0.25, [
                (0.6,1.0),(0.3,0.8),(0.5,0.9),
                (0.0,0.15),(0.3,0.8),(0.4,0.8),(0.05,0.2)
            ]),
            'io_bound'   : (0.20, [
                (0.1,0.4),(0.1,0.5),(0.1,0.4),
                (0.3,0.7),(0.1,0.4),(0.3,0.7),(0.4,0.8)
            ]),
            'mixed'      : (0.15, [
                (0.2,0.6),(0.1,0.4),(0.2,0.5),
                (0.3,0.6),(0.1,0.4),(0.3,0.6),(0.2,0.5)
            ]),
        }

        probs   = [v[0] for v in profiles.values()]
        choices = list(profiles.keys())

        for _ in range(n):
            profile = np.random.choice(choices, p=probs)
            ranges  = profiles[profile][1]
            obs     = [
                float(np.clip(
                    np.random.uniform(lo, hi), 0, 1))
                for lo, hi in ranges
            ]
            data.append(obs)
        return data

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx     = 0
        self.prev_wait       = 0.0
        self.episode_actions = []
        self.episode_rewards = []
        return self.snapshots[0], {}

    def step(self, action):
        obs    = self.snapshots[self.current_idx]
        reward = self._compute_reward(action, obs)

        self.episode_actions.append(int(action))
        self.episode_rewards.append(reward)

        self.current_idx += 1
        done = self.current_idx >= len(self.snapshots) - 1
        if done:
            self.current_idx = 0

        next_obs       = self.snapshots[self.current_idx]
        self.prev_wait = float(obs[1])

        return next_obs, reward, done, False, {
            'algorithm': self.algorithm_names[action],
            'wait'     : float(obs[1]),
            'cpu_load' : float(obs[0]),
        }

    def _compute_reward(self, action, obs):
        """
        Reward function aligned with eBPF features.
        Uses same feature indices as get_rl_observation().
        """
        cpu_bound   = obs[0]   # cpu_bound_ratio
        wait_norm   = obs[1]   # avg_wait_norm
        preempt     = obs[2]   # preempt_ratio
        interactive = obs[3]   # interactive_ratio
        load        = obs[4]   # load_norm
        queue       = obs[5]   # queue_length_norm
        migration   = obs[6]   # migration_ratio

        wait_improvement = self.prev_wait - wait_norm
        base = wait_improvement * 10.0

        bonus = 0.0
        if action == 0:    # FCFS
            # Good: low load, low queue, batch-like
            if load < 0.15 and interactive < 0.2:
                bonus = 0.6
            elif cpu_bound > 0.6:
                bonus = -0.4
        elif action == 1:  # SJF
            # Good: predictable bursts, moderate interactive
            if preempt < 0.3 and interactive > 0.3:
                bonus = 0.7
            elif cpu_bound > 0.7:
                bonus = -0.3
        elif action == 2:  # RR
            # Good: interactive, mixed, migration happening
            if interactive > 0.5 or migration > 0.3:
                bonus = 0.8
            elif load < 0.05:
                bonus = -0.1
        elif action == 3:  # PRIORITY
            # Good: high load, high preemption, cpu-bound
            if cpu_bound > 0.5 or preempt > 0.5:
                bonus = 0.9
            elif interactive > 0.8:
                bonus = -0.3

        penalty = -wait_norm * 2.0
        total   = base + bonus + penalty
        return float(np.clip(total, -2.0, 2.0))

    def get_episode_summary(self):
        if not self.episode_rewards:
            return {}
        return {
            'total_reward': sum(self.episode_rewards),
            'avg_reward'  : np.mean(self.episode_rewards),
            'fcfs'        : self.episode_actions.count(0),
            'sjf'         : self.episode_actions.count(1),
            'rr'          : self.episode_actions.count(2),
            'priority'    : self.episode_actions.count(3),
        }
