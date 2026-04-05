import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# ── Load and aggregate correctly ─────────────────────────────────
df = pd.read_csv('/root/AIMOS/data/raw_metrics.csv')
df['time_bucket'] = df['uptime_sec'].round(0)

agg = df.groupby('time_bucket').agg(
    cpu_mean      = ('cpu_percent',  'mean'),
    cpu_max       = ('cpu_percent',  'max'),
    cpu_bound_cnt = ('cpu_percent',  lambda x: (x > 10).sum()),
    mem_percent   = ('mem_percent',  'first'),
    avg_wait_ns   = ('wait_time_ns', 'mean'),
    num_procs     = ('pid',          'nunique'),
    disk_reads    = ('disk_reads',   'first'),
    disk_writes   = ('disk_writes',  'first'),
).reset_index().fillna(0)

print(f"Training snapshots: {len(agg)}")

# ── Normalise ─────────────────────────────────────────────────────
def norm(col, vmax=None):
    vmax = vmax or (col.max() + 1e-9)
    return (col / vmax).clip(0, 1)

agg['cpu_mean_n']  = norm(agg['cpu_mean'],      100)
agg['cpu_max_n']   = norm(agg['cpu_max'],        100)
agg['cpu_bound_n'] = norm(agg['cpu_bound_cnt'],  20)
agg['mem_n']       = norm(agg['mem_percent'],    100)
agg['wait_n']      = norm(agg['avg_wait_ns'])
agg['procs_n']     = norm(agg['num_procs'],      200)
agg['disk_n']      = norm(agg['disk_reads'] + agg['disk_writes'])

OBS_COLS   = ['cpu_mean_n','cpu_max_n','cpu_bound_n',
              'mem_n','wait_n','procs_n','disk_n']
data_array = agg[OBS_COLS].values.astype(np.float32)

print(f"Observation shape: {data_array.shape}")
print(f"\nFeature means:")
for col, val in zip(OBS_COLS, data_array.mean(axis=0)):
    bar = '█' * int(val * 20)
    print(f"  {col:15s}: {val:.4f}  {bar}")

# ── Environment ───────────────────────────────────────────────────
class CPUSchedulerEnv(gym.Env):
    """
    Actions: 0=powersave  1=ondemand  2=performance
    Reward designed from your real data ranges
    """
    metadata = {'render_modes': []}

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.idx  = 0
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = np.random.randint(0, len(self.data))
        return self.data[self.idx], {}

    def step(self, action):
        obs    = self.data[self.idx]
        reward = self._reward(obs, action)
        self.idx = (self.idx + 1) % len(self.data)
        return self.data[self.idx], reward, False, False, {}

    def _reward(self, obs, action):
        cpu  = float(obs[0])  # cpu_mean_n
        mem  = float(obs[3])  # mem_n
        wait = float(obs[4])  # wait_n
        disk = float(obs[6])  # disk_n

        # High CPU load → performance is correct
        if cpu > 0.6:
            if action == 2: return  2.0
            if action == 1: return  0.5
            if action == 0: return -2.0

        # Low CPU + low memory → powersave is correct
        if cpu < 0.2 and mem < 0.4:
            if action == 0: return  1.5
            if action == 1: return  0.2
            if action == 2: return -1.0

        # Medium load → ondemand is correct
        if 0.2 <= cpu <= 0.6:
            if action == 1: return  1.0
            if action == 2: return  0.3
            if action == 0: return -0.5

        # Heavy disk I/O → stay responsive
        if disk > 0.5 and action == 0:
            return -0.8

        # High wait time → don't use powersave
        if wait > 0.5 and action == 0:
            return -0.5

        return 0.1

# ── Train ─────────────────────────────────────────────────────────
os.makedirs('/root/AIMOS/models', exist_ok=True)

env      = Monitor(CPUSchedulerEnv(data_array),
                   '/root/AIMOS/logs/cpu_scheduler_train')
eval_env = Monitor(CPUSchedulerEnv(data_array),
                   '/root/AIMOS/logs/cpu_scheduler_eval')

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = '/root/AIMOS/models/',
    log_path             = '/root/AIMOS/logs/',
    eval_freq            = 1000,
    n_eval_episodes      = 20,
    deterministic        = True,
    verbose              = 1
)

model = PPO(
    'MlpPolicy', env,
    learning_rate = 3e-4,
    n_steps       = 128,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.95,
    ent_coef      = 0.01,   # encourages exploration
    verbose       = 1,
    tensorboard_log = '/root/AIMOS/logs/'
)

print("\nTraining PPO on your real kernel data...")
print("Watch explained_variance — it should rise toward 1.0\n")
model.learn(total_timesteps=100_000, callback=eval_cb)
model.save('/root/AIMOS/models/cpu_agent_ppo')
print("\nSaved → /root/AIMOS/models/cpu_agent_ppo.zip")

# ── Evaluate ──────────────────────────────────────────────────────
print("\n--- Agent decision distribution on your data ---")
labels  = {0:'powersave', 1:'ondemand', 2:'performance'}
counts  = {0:0, 1:0, 2:0}
rewards = []

obs, _ = eval_env.reset()
for _ in range(len(data_array)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = eval_env.step(int(action))
    counts[int(action)] += 1
    rewards.append(reward)

total = sum(counts.values())
for a, c in counts.items():
    pct = c / total * 100
    bar = '█' * int(pct / 3)
    print(f"  {labels[a]:12s}: {c:4d}  ({pct:5.1f}%)  {bar}")

print(f"\n  Mean reward per step: {np.mean(rewards):.4f}")
print(f"  Total reward:         {np.sum(rewards):.2f}")
print("\nNext step: run benchmark_compare.py to generate Table 3")
