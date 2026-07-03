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
        description="JAX-native curriculum training for KRTI gate navigation."
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which curriculum stage to start from (default: 1). "
             "Stages before this are skipped.",
    )
    parser.add_argument(
        "--restore-checkpoint",
        type=str,
        default=None,
        help="Path to a saved Brax checkpoint to restore params from before "
             "starting --start-stage. If omitted, training starts from scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Launching Overhauled JAX-Native Curriculum Gate Navigation")
    print("=" * 60)
    if args.start_stage > 1:
        print(f"[RESUME] Starting from curriculum stage {args.start_stage}")
    if args.restore_checkpoint:
        print(f"[RESUME] Restoring params from: {args.restore_checkpoint}")

    # Register the environment with Brax
    envs.register_environment('krti_gate_jax', KRTIAviaryJax)

    output_directory = "./results/krti_single_rl_jax/"
    os.makedirs(output_directory, exist_ok=True)
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    
    # Setup TensorBoard log directory
    tb_dir = os.path.join(output_directory, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(tb_dir, f"run_{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(tb_dir, f"run_{run_idx}")
    
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(run_dir)
    print(f"Logging TensorBoard events to: {run_dir}")


    curriculum_stages = [
        {"level": 1, "steps": 12_000_000, "lr": 3.0e-3, "entropy": 2.0e-2},  # Fixed config, high exploration
        {"level": 2, "steps": 12_000_000, "lr": 2.5e-3, "entropy": 1.5e-2},  # Minor variations
        {"level": 3, "steps": 12_000_000, "lr": 2.0e-4, "entropy": 1.0e-2},  # Moderate variations, speed scaling
        {"level": 4, "steps": 12_000_000, "lr": 1.5e-4, "entropy": 7.5e-3},  # Camera noise, aggressive offset
        {"level": 5, "steps": 12_000_000, "lr": 1.0e-4, "entropy": 5.0e-3},  # Full domain randomization
    ]

    global_step_counter = 0
    # Restore from a checkpoint if one was provided (e.g. when restarting a stage)
    restore_params = None
    if args.restore_checkpoint:
        print(f"Loading restore checkpoint: {args.restore_checkpoint}")
        restore_params = model.load_params(args.restore_checkpoint)

    print(f"Training on device: {jax.devices()[0]}")


    for stage_idx, stage in enumerate(curriculum_stages):
        level = stage["level"]
        steps = stage["steps"]
        lr = stage["lr"]
        entropy = stage["entropy"]

        # Skip stages that precede --start-stage
        if level < args.start_stage:
            global_step_counter += steps
            continue

        print(f"\n" + "#" * 60)
        print(f"STARTING CURRICULUM STAGE {level} ({steps} steps, LR: {lr})")
        print("#" * 60)

        # Initialize environment corresponding to this stage
        env = KRTIAviaryJax(curriculum_level=level)

        # Checkpoint saving callback function
        def save_checkpoint(current_step, make_policy, params):
            checkpoint_path = os.path.join(checkpoint_directory, f"checkpoint_stage_{level}_{current_step}")
            model.save_params(checkpoint_path, params)
            print(f" -> Checkpoint saved: Stage {level} step {current_step}")

        # Progress tracking callback function (Point 10)
        def progress(num_steps, metrics):
            global_steps = global_step_counter + num_steps
            print(f"Stage {level} - Step {num_steps} (Global: {global_steps}) - Reward: {metrics['eval/episode_reward']:.2f}")
            for name, value in metrics.items():
                tb_writer.add_scalar(name, float(value), global_steps)


        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=steps,
            num_evals=30,
            reward_scaling=0.1,            # Compress reward range to stabilize PPO critic
            episode_length=450,           
            normalize_observations=True,  # Crucial stabilizing feature for JAX environments
            action_repeat=1,
            
            # Aligned Parallelism Configs
            num_envs=32768,               # Doubled parallel environments (VRAM up)
            unroll_length=512,           # Keeps a strong 256-step trajectory horizon
            num_minibatches=64,         # (4096 * 256) / 512 = 2,048 steps per minibatch
            num_updates_per_batch=8,     # Standard PPO sweet-spot for deep learning utility
            
            discounting=0.99,
            learning_rate=lr,
            entropy_cost=entropy,         # Annealed per curriculum stage (2e-2 → 5e-3)
            
            seed=0,
            progress_fn=progress,
            policy_params_fn=save_checkpoint,
            restore_params=restore_params
        )

        # Save stage final parameters and hot-start next level
        stage_final_path = os.path.join(output_directory, f"stage_{level}_final")
        model.save_params(stage_final_path, params)
        print(f"Saved optimized curriculum level {level} parameters.")

        print("\n" + "=" * 60)
        print(f"STAGE {level} TRAINING COMPLETED SUCCESSFULLY.")
        print(f"Parameters saved to: {stage_final_path}")
        print("The training process is now paused.")
        print("You can run your evaluation script using the saved weights above.")
        print("=" * 60)

        # Loop until a valid choice is entered to avoid accidental aborts
        while True:
            choice = input(
                f"\nWould you like to advance to the next curriculum stage?\n"
                f"  [y]  Yes   — continue to stage {level + 1}\n"
                f"  [r]  Run   — evaluate stage {level} checkpoint (enjoy_jax.py)\n"
                f"  [s]  Stage — restart stage {level} from scratch (reloads updated code, uses XLA cache)\n"
                f"  [n]  No    — exit training\n"
                f"Choice: "
            ).strip().lower()

            if choice in ['y', 'yes']:
                print(f"\nConfirmed! Resuming execution and initializing curriculum level {level + 1}...")
                break
            elif choice in ['r', 'run', 'eval']:
                print(f"\nLaunching evaluation with checkpoint: {stage_final_path}")
                subprocess.run(
                    [
                        "python", "enjoy_jax.py",
                        "--model-path", stage_final_path,
                        "--curriculum-level", str(level),
                        "--steps", "600",
                    ],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                print("\nEvaluation finished. Returning to training prompt...")
            elif choice in ['s', 'stage', 'restart']:
                print(f"\nRestarting stage {level} from scratch with updated code...")
                print(f"   XLA cache: {_jax_cache}")
                print("   (Recompilation skipped if JAX-traced code is unchanged)")
                subprocess.Popen(
                    [
                        sys.executable, os.path.abspath(__file__),
                        "--start-stage", str(level),
                    ],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                print("New training process launched. Exiting current process.")
                sys.exit(0)
            elif choice in ['n', 'no', 'q', 'quit']:
                print(f"\nExiting training loop as requested. Your progress up to stage {level} is safely preserved.")
                sys.exit(0)
            else:
                print("Invalid response. Please enter 'y', 'r', 's', or 'n'.")

        restore_params = params
        global_step_counter += steps


    output_path = os.path.join(output_directory, "final_curriculum_policy")
    model.save_params(output_path, restore_params)
    print(f"\nCurriculum complete. Finalised parameters saved to {output_path}")

if __name__ == "__main__":
    main()