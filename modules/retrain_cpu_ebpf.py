"""
AIMOS — Retrain CPU Scheduler on eBPF observations.
Closes Layer 3 gap completely.
"""

import os
import sys
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback
)

from modules.cpu_scheduler_env_ebpf import CPUSchedulerEnvEBPF
from utils.config import (
    CPU_MODEL_PATH, CPU_ALGORITHMS,
    RESULT_DIR, LOG_DIR
)


def retrain():
    print("\n" + "="*55)
    print("  AIMOS — CPU Scheduler Retrain (eBPF data)")
    print("="*55 + "\n")

    os.makedirs(CPU_MODEL_PATH, exist_ok=True)

    train_env = Monitor(
        CPUSchedulerEnvEBPF(),
        filename=os.path.join(LOG_DIR, 'cpu_ebpf_train')
    )
    eval_env = Monitor(
        CPUSchedulerEnvEBPF(),
        filename=os.path.join(LOG_DIR, 'cpu_ebpf_eval')
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = CPU_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = 2000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1
    )

    ckpt_cb = CheckpointCallback(
        save_freq   = 10000,
        save_path   = CPU_MODEL_PATH,
        name_prefix = 'aimos_cpu_ebpf'
    )

    model = PPO(
        policy        = 'MlpPolicy',
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 256,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        verbose       = 1,
        device        = 'cpu'
    )

    print("Training PPO on eBPF-sourced observations...\n")
    model.learn(
        total_timesteps = 50_000,
        callback        = [eval_cb, ckpt_cb],
        progress_bar    = True
    )

    # Save as final model — overwrites previous
    final_path = os.path.join(CPU_MODEL_PATH, 'final_model')
    model.save(final_path)
    print(f"\nModel saved → {final_path}")

    # Evaluate
    print("\n--- Evaluating retrained agent ---")
    all_actions  = []
    all_rewards  = []
    env = CPUSchedulerEnvEBPF()

    for ep in range(10):
        obs, _ = env.reset()
        ep_r   = 0
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(int(action))
            ep_r += r
            all_actions.append(int(action))
        all_rewards.append(ep_r)

    print(f"\nAverage reward : {np.mean(all_rewards):.4f}")
    print(f"\nAlgorithm selection:")
    total = len(all_actions)
    for i, name in enumerate(CPU_ALGORITHMS):
        count = all_actions.count(i)
        pct   = 100 * count / total if total > 0 else 0
        bar   = '█' * int(pct / 2)
        print(f"  {name:<10}: {pct:5.1f}%  {bar}")

    # Save comparison
    results = {
        'algorithm': [CPU_ALGORITHMS[i] for i in all_actions],
    }
    pd.DataFrame(results).to_csv(
        os.path.join(RESULT_DIR, 'cpu_ebpf_actions.csv'),
        index=False
    )

    # Plot
    counts = [all_actions.count(i)
              for i in range(len(CPU_ALGORITHMS))]
    plt.figure(figsize=(8, 4))
    plt.bar(CPU_ALGORITHMS, counts,
            color=['#3B82F6','#10B981','#F59E0B','#EF4444'])
    plt.title('AIMOS — CPU Scheduler (eBPF-trained)')
    plt.ylabel('Times selected')
    plt.tight_layout()
    path = os.path.join(
        RESULT_DIR, 'cpu_ebpf_distribution.png'
    )
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Chart saved → {path}")

    print(f"\n{'='*55}")
    print(f"  Fix 1 COMPLETE — Layer 3 gap closed")
    print(f"  Agent retrained on eBPF kernel observations")
    print(f"{'='*55}\n")
    return model


if __name__ == '__main__':
    retrain()
