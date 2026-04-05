"""
AIMOS — Module 6: Energy-Aware OS Optimizer
Multi-objective RL that balances performance vs power consumption.
Controls CPU frequency governors on Linux via /sys/devices/system/cpu.

Actions (CPU governor policies):
  0 = performance   — maximum frequency always
  1 = balanced      — balance speed and power
  2 = powersave     — minimum frequency
  3 = adaptive      — AI decides per-workload
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from utils.config import (
    ENERGY_MODEL_PATH,
    ENERGY_LAMBDA,
    ENERGY_TRAIN_STEPS,
    RAW_METRICS_CSV,
    RESULT_DIR,
    LOG_DIR
)

ENERGY_GOVERNORS = ['performance', 'balanced', 'powersave', 'adaptive']


class EnergyOptimizerEnv(gym.Env):
    """
    Multi-objective RL environment for energy-aware CPU management.

    State:
      - cpu_percent_norm    : current CPU utilization
      - mem_percent_norm    : memory pressure
      - load_avg_norm       : system load
      - active_procs_norm   : number of active processes
      - io_pressure_norm    : I/O activity level
      - time_of_day_norm    : time-based usage pattern

    Action:
      0-3 : CPU governor selection

    Reward:
      R = performance_score - lambda * energy_cost
      lambda (ENERGY_LAMBDA) controls the trade-off
    """

    def __init__(self, data_path=RAW_METRICS_CSV, lam=ENERGY_LAMBDA):
        super().__init__()
        self.lam         = lam
        self.data_path   = data_path
        self.current_idx = 0

        self.action_space = spaces.Discrete(len(ENERGY_GOVERNORS))
        self.observation_space = spaces.Box(
            low   = np.zeros(6, dtype=np.float32),
            high  = np.ones(6,  dtype=np.float32),
            dtype = np.float32
        )

        self._load_data()
        self.governor_history = []

    def _load_data(self):
        """Load system-level snapshots for energy optimization."""
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            sys_df = df[df['type'] == 'system'].copy()
            num_cols = sys_df.select_dtypes(include='number').columns
            str_cols = sys_df.select_dtypes(include='object').columns
            sys_df[num_cols] = sys_df[num_cols].fillna(0)
            sys_df[str_cols] = sys_df[str_cols].fillna('')
            if len(sys_df) >= 10:
                self.snapshots = self._build_state_matrix(sys_df)
                print(f"[Module 6] Loaded {len(self.snapshots)} system snapshots")
                return
        print("[Module 6] Generating synthetic energy data")
        self.snapshots = self._generate_synthetic(400)

    def _build_state_matrix(self, sys_df):
        states = []
        for _, row in sys_df.iterrows():
            state = [
                np.clip(row.get('cpu_percent', 0)   / 100.0, 0, 1),
                np.clip(row.get('mem_percent',  0)  / 100.0, 0, 1),
                np.clip(row.get('load_avg_1m',  0)  / 8.0,   0, 1),
                np.clip(row.get('disk_io_active', 0)/ 10.0,  0, 1),
                np.clip(row.get('swap_percent', 0)  / 100.0, 0, 1),
                float(np.random.uniform(0, 1)),  # time of day proxy
            ]
            states.append([float(np.clip(x, 0, 1)) for x in state])
        return np.array(states, dtype=np.float32)

    def _generate_synthetic(self, n=400):
        np.random.seed(42)
        states = []
        for _ in range(n):
            profile = np.random.choice(
                ['idle','light','moderate','heavy'],
                p=[0.2, 0.3, 0.3, 0.2]
            )
            if profile == 'idle':
                s = [
                    np.random.uniform(0.0, 0.08),
                    np.random.uniform(0.1, 0.3),
                    np.random.uniform(0.0, 0.05),
                    np.random.uniform(0.0, 0.02),
                    np.random.uniform(0.0, 0.01),
                    np.random.uniform(0.0, 1.0),
                ]
            elif profile == 'light':
                s = [
                    np.random.uniform(0.05, 0.25),
                    np.random.uniform(0.2,  0.5),
                    np.random.uniform(0.05, 0.2),
                    np.random.uniform(0.0,  0.1),
                    np.random.uniform(0.0,  0.05),
                    np.random.uniform(0.0,  1.0),
                ]
            elif profile == 'moderate':
                s = [
                    np.random.uniform(0.25, 0.6),
                    np.random.uniform(0.4,  0.7),
                    np.random.uniform(0.2,  0.5),
                    np.random.uniform(0.1,  0.4),
                    np.random.uniform(0.0,  0.1),
                    np.random.uniform(0.0,  1.0),
                ]
            else:  # heavy
                s = [
                    np.random.uniform(0.6,  1.0),
                    np.random.uniform(0.6,  0.95),
                    np.random.uniform(0.5,  1.0),
                    np.random.uniform(0.3,  0.8),
                    np.random.uniform(0.0,  0.3),
                    np.random.uniform(0.0,  1.0),
                ]
            states.append([float(np.clip(x, 0, 1)) for x in s])
        return np.array(states, dtype=np.float32)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx  = 0
        self.governor_history = []
        return self.snapshots[0], {}

    def step(self, action):
        obs     = self.snapshots[self.current_idx]
        reward  = self._compute_reward(action, obs)

        self.governor_history.append({
            'action'     : int(action),
            'governor'   : ENERGY_GOVERNORS[action],
            'cpu'        : float(obs[0]),
            'reward'     : reward,
        })

        self.current_idx += 1
        done = self.current_idx >= len(self.snapshots) - 1
        if done:
            self.current_idx = 0

        next_obs = self.snapshots[self.current_idx]
        return next_obs, reward, done, False, {
            'governor'   : ENERGY_GOVERNORS[action],
            'cpu_load'   : float(obs[0]),
            'reward'     : reward,
        }

    def _compute_reward(self, action, obs):
        """
        Multi-objective reward:
          R = performance_score - lambda * energy_cost

        Key logic:
          performance = high cpu → needs fast governor
          powersave   = low  cpu → save energy
          balanced    = medium load → middle ground
          adaptive    = AI bonus for context-aware decision
        """
        cpu   = obs[0]
        mem   = obs[1]
        load  = obs[2]
        io    = obs[3]

        # Performance score — how well we serve the workload
        if action == 0:    # performance governor
            perf_score = cpu * 1.0 + load * 0.5
            if cpu < 0.1:
                perf_score -= 0.4   # wasteful under idle load
        elif action == 1:  # balanced
            perf_score = cpu * 0.7 + load * 0.4
        elif action == 2:  # powersave
            perf_score = cpu * 0.3 + load * 0.2
            if cpu > 0.5:
                perf_score -= 0.5   # severely hurts heavy workloads
        else:              # adaptive — context-aware bonus
            if cpu > 0.6:
                perf_score = cpu * 0.9   # near-performance under load
            elif cpu < 0.15:
                perf_score = 0.3         # near-powersave when idle
            else:
                perf_score = cpu * 0.7   # balanced middle ground

        # Energy cost — higher frequency = more power
        energy_costs = {
            0: cpu * 1.0  + 0.3,       # performance: always high cost
            1: cpu * 0.6  + 0.15,      # balanced: moderate cost
            2: cpu * 0.2  + 0.05,      # powersave: minimal cost
            3: cpu * 0.5  + 0.1,       # adaptive: smart cost
        }
        energy_cost = energy_costs[action]

        # Multi-objective reward
        reward = perf_score - self.lam * energy_cost
        return float(np.clip(reward, -1.5, 1.5))

    def get_pareto_analysis(self):
        """Return performance vs energy trade-off data for paper."""
        if not self.governor_history:
            return pd.DataFrame()
        return pd.DataFrame(self.governor_history)


def train_energy_optimizer():
    print("\n" + "="*55)
    print("  AIMOS — Module 6: Energy Optimizer")
    print("="*55 + "\n")

    os.makedirs(ENERGY_MODEL_PATH, exist_ok=True)

    train_env = Monitor(EnergyOptimizerEnv(),
                        filename=os.path.join(LOG_DIR, 'energy_train'))
    eval_env  = Monitor(EnergyOptimizerEnv(),
                        filename=os.path.join(LOG_DIR, 'energy_eval'))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = ENERGY_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = 2000,
        n_eval_episodes      = 5,
        verbose              = 1
    )

    model = PPO(
        policy        = 'MlpPolicy',
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 256,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        verbose       = 1,
        device        = 'cpu'
    )

    print(f"Training for {ENERGY_TRAIN_STEPS:,} timesteps...\n")
    model.learn(
        total_timesteps = ENERGY_TRAIN_STEPS,
        callback        = eval_cb,
        progress_bar    = True
    )

    final_path = os.path.join(ENERGY_MODEL_PATH, 'final_model')
    model.save(final_path)
    print(f"\nModel saved → {final_path}")

    # ── Evaluate and compare governors ───────────────────────────────────
    print("\n--- Governor comparison (RL agent vs fixed policies) ---")
    _compare_governors(model)
    _plot_pareto_front()

    print(f"\n{'='*55}")
    print(f"  Module 6 COMPLETE")
    print(f"{'='*55}\n")

    return model


def _compare_governors(model):
    """Compare RL agent against fixed governor baselines."""
    env     = EnergyOptimizerEnv()
    results = {}

    # Fixed governor baselines
    for gov_id, gov_name in enumerate(ENERGY_GOVERNORS[:-1]):
        rewards = []
        for _ in range(5):
            obs, _ = env.reset()
            ep_r   = 0
            done   = False
            while not done:
                obs, r, done, _, _ = env.step(gov_id)
                ep_r += r
            rewards.append(ep_r)
        results[gov_name] = np.mean(rewards)

    # RL agent
    rl_rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        ep_r   = 0
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(int(action))
            ep_r += r
        rl_rewards.append(ep_r)
    results['AIMOS-RL'] = np.mean(rl_rewards)

    print(f"\n  {'Governor':<14} {'Avg Reward':>12}")
    print(f"  {'-'*28}")
    for name, val in sorted(results.items(),
                            key=lambda x: x[1], reverse=True):
        marker = ' ← AIMOS' if name == 'AIMOS-RL' else ''
        print(f"  {name:<14} {val:>12.4f}{marker}")

    pd.DataFrame(
        list(results.items()),
        columns=['governor', 'avg_reward']
    ).to_csv(
        os.path.join(RESULT_DIR, 'energy_governor_comparison.csv'),
        index=False
    )
    print(f"\n  Results saved → {RESULT_DIR}/energy_governor_comparison.csv")


def _plot_pareto_front():
    """Plot performance vs energy trade-off curve."""
    lambdas = np.linspace(0.0, 1.0, 20)
    perf_scores  = []
    energy_scores = []

    for lam in lambdas:
        env = EnergyOptimizerEnv(lam=lam)
        obs, _ = env.reset()
        total_perf   = 0
        total_energy = 0

        for _ in range(100):
            # Simulate adaptive governor
            cpu = obs[0]
            if cpu > 0.6:
                action = 0   # performance
            elif cpu < 0.15:
                action = 2   # powersave
            else:
                action = 1   # balanced

            obs, r, done, _, _ = env.step(action)
            total_perf   += obs[0]
            total_energy += (1 - obs[0])
            if done:
                break

        perf_scores.append(total_perf / 100)
        energy_scores.append(total_energy / 100)

    plt.figure(figsize=(8, 5))
    plt.plot(energy_scores, perf_scores,
             'o-', color='#3B82F6', linewidth=2, markersize=6)
    plt.xlabel('Energy consumption (normalized)')
    plt.ylabel('Performance score (normalized)')
    plt.title('AIMOS — Pareto front: Performance vs Energy trade-off')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, 'energy_pareto_front.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Pareto front plot saved → {path}")


if __name__ == '__main__':
    train_energy_optimizer()
