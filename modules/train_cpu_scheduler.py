"""
AIMOS — Module 1: CPU Scheduler RL Training
Trains a PPO agent to select optimal scheduling algorithms
based on real OS workload patterns.

Uses Stable-Baselines3 PPO — more stable than raw Q-Learning
and better suited to the continuous observation space.
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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

from modules.cpu_scheduler_env import CPUSchedulerEnv
from utils.config import (
    CPU_MODEL_PATH,
    CPU_TRAIN_TIMESTEPS,
    CPU_ALGORITHMS,
    RESULT_DIR,
    LOG_DIR
)


def train():
    print("\n" + "="*55)
    print("  AIMOS — CPU Scheduler RL Training")
    print("="*55 + "\n")

    os.makedirs(CPU_MODEL_PATH, exist_ok=True)
    os.makedirs(RESULT_DIR,     exist_ok=True)

    # ── Create environments ───────────────────────────────────────────────
    train_env = Monitor(
        CPUSchedulerEnv(),
        filename=os.path.join(LOG_DIR, 'cpu_scheduler_train')
    )
    eval_env = Monitor(
        CPUSchedulerEnv(),
        filename=os.path.join(LOG_DIR, 'cpu_scheduler_eval')
    )

    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")
    print(f"Algorithms        : {CPU_ALGORITHMS}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = CPU_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = 2000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq = 10000,
        save_path = CPU_MODEL_PATH,
        name_prefix = 'aimos_cpu_rl'
    )

    # ── PPO Agent ─────────────────────────────────────────────────────────
    model = PPO(
        policy         = 'MlpPolicy',
        env            = train_env,
        learning_rate  = 3e-4,
        n_steps        = 256,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        verbose        = 1,
        tensorboard_log= LOG_DIR,
        device         = 'cpu'
    )

    print(f"Training for {CPU_TRAIN_TIMESTEPS:,} timesteps...\n")

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = CPU_TRAIN_TIMESTEPS,
        callback        = [eval_callback, checkpoint_callback],
        progress_bar    = True
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(CPU_MODEL_PATH, 'final_model')
    model.save(final_path)
    print(f"\nModel saved → {final_path}")

    # ── Evaluate trained agent ─────────────────────────────────────────────
    print("\n--- Evaluating trained agent ---")
    evaluate_agent(model, n_episodes=10)

    # ── Compare with baselines ────────────────────────────────────────────
    print("\n--- Baseline comparison ---")
    compare_with_baselines()

    return model


def evaluate_agent(model, n_episodes=10):
    """Run trained agent and print per-algorithm statistics."""
    env = CPUSchedulerEnv()
    all_rewards  = []
    all_actions  = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            ep_reward += reward
            all_actions.append(int(action))

        all_rewards.append(ep_reward)
        summary = env.get_episode_summary()

    # Algorithm selection distribution
    total = len(all_actions)
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"  Average reward     : {np.mean(all_rewards):.3f}")
    print(f"  Std reward         : {np.std(all_rewards):.3f}")
    print(f"\nAlgorithm selection distribution:")
    for i, name in enumerate(CPU_ALGORITHMS):
        count = all_actions.count(i)
        pct   = 100 * count / total if total > 0 else 0
        bar   = '█' * int(pct / 2)
        print(f"  {name:<10}: {pct:5.1f}%  {bar}")

    # Save results
    results = {
        'algorithm'  : [CPU_ALGORITHMS[i] for i in all_actions],
        'action_id'  : all_actions,
    }
    pd.DataFrame(results).to_csv(
        os.path.join(RESULT_DIR, 'cpu_scheduler_actions.csv'),
        index=False
    )
    print(f"\nAction log saved → {RESULT_DIR}/cpu_scheduler_actions.csv")

    # Plot
    _plot_algorithm_distribution(all_actions)
    _plot_rewards(all_rewards)


def compare_with_baselines():
    """
    Compare RL agent against static algorithm baselines.
    This is your results table for the paper.
    """
    env = CPUSchedulerEnv()

    results = {}

    # Fixed baselines — always pick the same algorithm
    for algo_id, algo_name in enumerate(CPU_ALGORITHMS):
        rewards = []
        waits   = []

        for _ in range(5):
            obs, _ = env.reset()
            ep_reward = 0
            done = False

            while not done:
                obs, reward, done, _, info = env.step(algo_id)
                ep_reward += reward
                waits.append(info['wait_time'])

            rewards.append(ep_reward)

        results[algo_name] = {
            'avg_reward'    : np.mean(rewards),
            'avg_wait_time' : np.mean(waits),
        }

    # Print comparison table
    print(f"\n{'Algorithm':<12} {'Avg Reward':>12} {'Avg Wait':>12}")
    print("-" * 38)
    for name, vals in results.items():
        print(f"{name:<12} {vals['avg_reward']:>12.4f} "
              f"{vals['avg_wait_time']:>12.6f}")

    # Save for paper
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(RESULT_DIR, 'cpu_baseline_comparison.csv'))
    print(f"\nBaseline table saved → "
          f"{RESULT_DIR}/cpu_baseline_comparison.csv")


def _plot_algorithm_distribution(actions):
    counts = [actions.count(i) for i in range(len(CPU_ALGORITHMS))]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(CPU_ALGORITHMS, counts,
                   color=['#3B82F6','#10B981','#F59E0B','#EF4444'])
    plt.title('AIMOS — Algorithm selection by RL agent')
    plt.ylabel('Times selected')
    plt.xlabel('Scheduling algorithm')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, 'cpu_algo_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Chart saved → {path}")


def _plot_rewards(rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o', color='#3B82F6', linewidth=2)
    plt.axhline(np.mean(rewards), color='red',
                linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.title('AIMOS — RL agent reward per episode')
    plt.ylabel('Total episode reward')
    plt.xlabel('Episode')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, 'cpu_rewards.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Chart saved → {path}")


if __name__ == '__main__':
    train()
