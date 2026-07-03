import os
# Force JAX to allocate memory dynamically as needed instead of pre-claiming 75%
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Limit compiler parallel threads to reduce peak host-RAM spikes during optimization
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

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


def main():
    print("=" * 60)
    print("Launching Overhauled JAX-Native Curriculum Gate Navigation")
    print("=" * 60)

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
        {"level": 1, "steps": 3_000_000, "lr": 3.0e-4}, # Fixed configuration, low speed
        {"level": 2, "steps": 3_000_000, "lr": 2.5e-4}, # Minor variations
        {"level": 3, "steps": 4_000_000, "lr": 2.0e-4}, # Moderate variations, speed scaling
        {"level": 4, "steps": 4_000_000, "lr": 1.5e-4}, # Camera noise, aggressive offset
        {"level": 5, "steps": 6_000_000, "lr": 1.0e-4}, # Full domain randomization
    ]

    global_step_counter = 0
    restore_params = None

    print(f"Training on device: {jax.devices()[0]}")


    for stage_idx, stage in enumerate(curriculum_stages):
        level = stage["level"]
        steps = stage["steps"]
        lr = stage["lr"]

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
            num_evals=15,
            reward_scaling=1.0,           
            episode_length=450,           
            normalize_observations=True,  # Crucial stabilizing feature for JAX environments
            action_repeat=1,
            
            # Aligned Parallelism Configs
            num_envs=2048,
            unroll_length=8,
            num_minibatches=256,
            num_updates_per_batch=4,
            
            discounting=0.99,
            learning_rate=lr,
            entropy_cost=1.0e-2,
            
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
                f"\nWould you like to advance to the next curriculum stage? [y]es / [n]o / [q]uit: "
            ).strip().lower()
            
            if choice in ['y', 'yes']:
                print(f"\nConfirmed! Resuming execution and initializing curriculum level {level + 1}...")
                break
            elif choice in ['n', 'no', 'q', 'quit']:
                print(f"\nExiting training loop as requested. Your progress up to stage {level} is safely preserved.")
                import sys
                sys.exit(0)
            else:
                print("Invalid response. Please enter 'y' to continue, or 'n' to stop training.")

        restore_params = params
        global_step_counter += steps


    output_path = os.path.join(output_directory, "final_curriculum_policy")
    model.save_params(output_path, restore_params)
    print(f"\nCurriculum complete. Finalised parameters saved to {output_path}")

if __name__ == "__main__":
    main()