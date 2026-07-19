#!/usr/bin/env python
import os
import sys
import argparse
import json
import time

# Force JAX to allocate memory dynamically as needed instead of pre-claiming 75%
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Limit compiler parallel threads to reduce peak host-RAM spikes during optimization
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import jax
import numpy as np

# Enable compilation caching to speed up warm restarts
_jax_cache = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"),
)
os.makedirs(_jax_cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache)

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model

from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax

try:
    import optuna
except ImportError:
    print("Error: optuna is not installed. Please install it first:")
    print("  pip install optuna optuna-dashboard")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweeps for JAX drone navigation (to optimize gate clearance & reduce crashes)."
    )
    parser.add_argument(
        "--curriculum-level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Which curriculum level / stage to tune parameters on (default: 3).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2_000_000,
        help="Number of training steps per Optuna trial (default: 2,000,000).",
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=10,
        help="Number of evaluations during training to check progress (default: 10).",
    )
    parser.add_argument(
        "--restore-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint parameters file to warm-start training from (recommended to restore from previous curriculum level).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Brax training (default: 42).",
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default="optuna_study.db",
        help="SQLite database file for Optuna tracking (default: optuna_study.db).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="best_hyperparameters.json",
        help="Output JSON file for the best hyperparameters (default: best_hyperparameters.json).",
    )
    parser.add_argument(
        "--tb-log",
        action="store_true",
        help="Enable TensorBoard logging for each trial run (stored under results/krti_single_rl_jax/optuna_tensorboard/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Optuna Hyperparameter Tuning for KRTI Gate Navigation JAX")
    print("=" * 60)
    print(f"Curriculum Level: {args.curriculum_level}")
    print(f"Steps per Trial:  {args.steps}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"SQLite DB File:   sqlite:///{args.db_file}")
    if args.restore_checkpoint:
        print(f"Restoring from:   {args.restore_checkpoint}")
    print("=" * 60)

    # Register the environment class
    envs.register_environment('krti_gate_jax', KRTIAviaryJax)

    # Set up the database storage URL
    db_url = f"sqlite:///{args.db_file}"

    # Define the objective function for Optuna
    def objective(trial):
        # 1. Sample Environment Reward & Penalty Weights (High Leverage)
        crash_penalty = trial.suggest_float("crash_penalty", -5000.0, -1000.0)
        centering_weight = trial.suggest_float("centering_weight", 1.0, 8.0)
        attitude_penalty_weight = trial.suggest_float("attitude_penalty_weight", -0.5, -0.05)
        smoothness_penalty_weight = trial.suggest_float("smoothness_penalty_weight", -0.05, -0.001)
        survival_reward = trial.suggest_float("survival_reward", 0.0, 0.2)

        # 2. Sample PPO Optimization Multipliers instead of absolute values
        entropy_multiplier = trial.suggest_float("entropy_multiplier", 0.5, 2.0)
        lr_multiplier = trial.suggest_float("lr_multiplier", 0.5, 2.0)
        discounting = trial.suggest_float("discounting", 0.98, 0.995)

        # Lookup the scheduled baseline for the current tuning stage
        # (Matches the curriculum_stages table in pod.py)
        base_schedule = {
            1: (3.0e-4, 1.0e-3),
            2: (2.5e-4, 5.0e-4),
            3: (2.0e-4, 5.0e-4),
            4: (1.0e-4, 1.0e-4),
            5: (2.0e-4, 1.0e-4)
        }
        base_lr, base_entropy = base_schedule.get(args.curriculum_level, (2.0e-4, 5.0e-4))

        effective_lr = base_lr * lr_multiplier
        effective_entropy = base_entropy * entropy_multiplier

        print(f"\n[Trial {trial.number}] Starting trial with parameters:")
        print(f"  crash_penalty:             {crash_penalty:.2f}")
        print(f"  centering_weight:          {centering_weight:.2f}")
        print(f"  attitude_penalty_weight:   {attitude_penalty_weight:.4f}")
        print(f"  smoothness_penalty_weight: {smoothness_penalty_weight:.5f}")
        print(f"  survival_reward:           {survival_reward:.4f}")
        print(f"  entropy_multiplier:        {entropy_multiplier:.4f} (Effective: {effective_entropy:.2e})")
        print(f"  discounting:               {discounting:.4f}")
        print(f"  lr_multiplier:             {lr_multiplier:.4f} (Effective: {effective_lr:.2e})")

        # Create the environment with the sampled parameters
        env = KRTIAviaryJax(
            curriculum_level=args.curriculum_level,
            crash_penalty=crash_penalty,
            centering_weight=centering_weight,
            attitude_penalty_weight=attitude_penalty_weight,
            smoothness_penalty_weight=smoothness_penalty_weight,
            survival_reward=survival_reward,
        )

        # Load restore checkpoint parameters if specified
        restore_params = None
        if args.restore_checkpoint:
            restore_params = model.load_params(args.restore_checkpoint)

        # Setup TensorBoard log directory for this trial if enabled
        tb_writer = None
        if args.tb_log:
            tb_dir = "./results/krti_single_rl_jax/optuna_tensorboard"
            trial_tb_dir = os.path.join(tb_dir, f"trial_{trial.number}")
            os.makedirs(trial_tb_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(trial_tb_dir)

        # We'll collect evaluation metrics to prune trials early and compute final score
        trial_metrics = []

        def progress_fn(num_steps, metrics):
            cleared_gate = float(metrics.get("eval/episode_cleared_gate", 0.0))
            crashed = float(metrics.get("eval/episode_crashed", 0.0))
            reward = float(metrics.get("eval/episode_reward", 0.0))
            gate_collided = float(metrics.get("eval/episode_gate_collided", 0.0))
            gate_distance = float(metrics.get("eval/episode_gate_distance", 0.0))
            smoothness = float(metrics.get("eval/episode_reward_smooth", 0.0))

            # The score we want to maximize: High clearance, zero crashes
            score = cleared_gate - crashed
            trial_metrics.append({
                "step": num_steps,
                "cleared_gate": cleared_gate,
                "crashed": crashed,
                "reward": reward,
                "gate_collided": gate_collided,
                "gate_distance": gate_distance,
                "smoothness": smoothness,
                "score": score
            })

            print(f"Stage {args.curriculum_level} - Step {num_steps} (Global: {num_steps}) - Reward: {reward:.2f} - Cleared: {cleared_gate:.4f} - Crashed: {crashed:.4f} -> Score: {score:.4f}")

            # Log to TensorBoard if enabled
            if tb_writer is not None:
                for name, value in metrics.items():
                    tb_writer.add_scalar(name, float(value), num_steps)

            # Report step score to Optuna for pruning
            trial.report(score, num_steps)

            # Check if this trial should be pruned based on early intermediate results
            if trial.should_prune():
                print(f"  [Trial {trial.number}] Pruning trial early...")
                if tb_writer is not None:
                    tb_writer.close()
                raise optuna.TrialPruned()

        try:
            # Train using Brax PPO and dynamic hyperparameters
            ppo.train(
                environment=env,
                num_timesteps=args.steps,
                num_evals=args.num_evals,
                reward_scaling=0.01,
                episode_length=450,
                normalize_observations=True,
                action_repeat=1,
                num_envs=1024,
                unroll_length=128,
                num_minibatches=32,
                num_updates_per_batch=4,
                discounting=discounting,
                learning_rate=effective_lr,
                entropy_cost=effective_entropy,
                seed=args.seed,
                progress_fn=progress_fn,
                restore_params=restore_params,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  [Trial {trial.number}] Error occurred during PPO training: {e}")
            if tb_writer is not None:
                tb_writer.close()
            return -2.0

        if not trial_metrics:
            return -2.0

        # Compute final score by averaging the last few evaluation points to avoid noisy trials
        last_evals = trial_metrics[-min(5, len(trial_metrics)):]
        avg_cleared = sum(x["cleared_gate"] for x in last_evals) / len(last_evals)
        avg_crashed = sum(x["crashed"] for x in last_evals) / len(last_evals)
        avg_reward = sum(x["reward"] for x in last_evals) / len(last_evals)
        avg_collided = sum(x["gate_collided"] for x in last_evals) / len(last_evals)
        avg_distance = sum(x["gate_distance"] for x in last_evals) / len(last_evals)
        avg_smoothness = sum(x["smoothness"] for x in last_evals) / len(last_evals)
        
        final_score = avg_cleared - avg_crashed

        # Save user-facing metrics as attributes for easier dashboard plotting
        trial.set_user_attr("final_cleared_gate", avg_cleared)
        trial.set_user_attr("final_crashed", avg_crashed)
        trial.set_user_attr("final_gate_collided", avg_collided)
        trial.set_user_attr("final_reward", avg_reward)
        trial.set_user_attr("final_gate_distance", avg_distance)
        trial.set_user_attr("final_smoothness", avg_smoothness)

        print(f"[Trial {trial.number}] Finished! Average Cleared Gate: {avg_cleared:.4f}, Crashed: {avg_crashed:.4f}, Reward: {avg_reward:.2f} -> Final Score: {final_score:.4f}")
        
        if tb_writer is not None:
            tb_writer.close()
            
        return final_score

    # Create or load the persistent Optuna study
    study = optuna.create_study(
        study_name="drone_gate_navigation",
        direction="maximize",
        storage=db_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    )

    print(f"Optuna database initialized")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY.")
    print("=" * 60)
    best_trial = study.best_trial
    print(f"Best Trial: Number {best_trial.number} with Score: {best_trial.value:.4f}")
    print("Best Parameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # Export best configurations
    config = {
        "best_score": best_trial.value,
        "best_trial_number": best_trial.number,
        "curriculum_level": args.curriculum_level,
        "parameters": best_trial.params,
        "metrics": best_trial.user_attrs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(args.output_config, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"\nConfiguration successfully saved to: {args.output_config}")
    print("\nTo launch the Optuna Dashboard and visualize results, run:")
    print(f"  optuna-dashboard sqlite:///{args.db_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
