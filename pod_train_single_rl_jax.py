import os
import sys
import argparse
import subprocess

# Force JAX to allocate memory dynamically as needed instead of pre-claiming 75%
# Limit compiler parallel threads to reduce peak host-RAM spikes during optimization
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import jax
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="JAX-native single-stage curriculum training for KRTI gate navigation."
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which curriculum stage to train (default: 1).",
    )
    parser.add_argument(
        "--restore-checkpoint",
        type=str,
        default=None,
        help="Path to a saved Brax checkpoint to restore params from. "
             "If omitted, it will try to auto-load the final model from the previous stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage = args.stage

    print("=" * 60)
    print(f"Launching JAX-Native Training for Curriculum Stage {stage}")
    print("=" * 60)

    # Register the environment with Brax
    envs.register_environment('krti_gate_jax', KRTIAviaryJax)

    output_directory = "./results/krti_single_rl_jax/"
    os.makedirs(output_directory, exist_ok=True)
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    
    # Setup TensorBoard log directory for this specific stage
    tb_dir = os.path.join(output_directory, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(tb_dir, f"run_stage_{stage}_{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(tb_dir, f"run_stage_{stage}_{run_idx}")
    
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(run_dir)
    print(f"Logging TensorBoard events to: {run_dir}")

    # Restore checkpoint loading
    restore_params = None
    if args.restore_checkpoint:
        print(f"[RESTORE] Loading custom restore checkpoint: {args.restore_checkpoint}")
        restore_params = model.load_params(args.restore_checkpoint)
    elif stage > 1:
        # Auto-restore from previous stage final parameter file
        prev_stage_path = os.path.join(output_directory, f"stage_{stage-1}_final")
        if os.path.exists(prev_stage_path):
            print(f"[AUTO-RESTORE] Found previous stage final model. Loading: {prev_stage_path}")
            restore_params = model.load_params(prev_stage_path)
        else:
            print(f"[WARNING] No checkpoint found for previous stage at {prev_stage_path}. Starting Stage {stage} from scratch!")

    print(f"Training on device: {jax.devices()[0]}")

    # Build stage environment
    env = KRTIAviaryJax(curriculum_level=stage)

    # Compile-cache functions import
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    
    custom_network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256)
    )

    stage_steps = 20_971_520  # 5 epochs of 4.19M steps
    stage_evals = 20          # Evaluate 20 times per stage

    def progress_fn(num_steps, metrics):
        print(f"[Stage {stage}] Step {num_steps} - Reward: {metrics['eval/episode_reward']:.2f}")
        for name, value in metrics.items():
            tb_writer.add_scalar(name, float(value), num_steps)

    def checkpoint_fn(current_step, make_policy, params):
        checkpoint_path = os.path.join(checkpoint_directory, f"checkpoint_stage{stage}_{current_step}")
        model.save_params(checkpoint_path, params)
        print(f" -> Checkpoint saved: Stage {stage} Step {current_step}")

    print(f"Training Stage {stage} for {stage_steps} steps (evals={stage_evals})...")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=stage_steps,
        num_evals=stage_evals,
        reward_scaling=1.0,
        episode_length=450,
        normalize_observations=True,
        action_repeat=1,
        network_factory=custom_network_factory,
        
        # PPO Hyperparameters
        num_envs=2048,
        unroll_length=128,
        batch_size=1024,
        num_minibatches=32,
        num_updates_per_batch=4,
        
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=0.01,
        
        seed=0,
        progress_fn=progress_fn,
        policy_params_fn=checkpoint_fn,
        restore_params=restore_params
    )

    # Save stage final parameters
    stage_final_path = os.path.join(output_directory, f"stage_{stage}_final")
    model.save_params(stage_final_path, params)
    print(f"[SUCCESS] Completed Stage {stage}. Parameters saved to {stage_final_path}")

    # Also duplicate final params to the subsequent stages for enjoy_jax compatibility
    for lvl in range(stage, 6):
        compat_path = os.path.join(output_directory, f"stage_{lvl}_final")
        model.save_params(compat_path, params)

    tb_writer.close()
    print("\n" + "=" * 60)
    print(f"CURRICULUM STAGE {stage} COMPLETED SUCCESSFULLY.")
    print("Parameters saved to results directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()