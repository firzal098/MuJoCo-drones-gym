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
    print("Launching MuJoCo MJX JAX-Native Training System")
    print("=" * 60)

    # Register the environment with Brax
    envs.register_environment('krti_gate_jax', KRTIAviaryJax)

    output_directory = "./results/krti_single_rl_jax/"
    os.makedirs(output_directory, exist_ok=True)
    model_path = "./results/krti_single_rl_jax/finalised/stage_1"
    output_path = "./results/krti_single_rl_jax/finalised/final"
    env = KRTIAviaryJax()
    
    # Find next incremental run directory index for Tensorboard (similar to SB3)
    tb_dir = os.path.join(output_directory, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(tb_dir, f"run_{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(tb_dir, f"run_{run_idx}")
    
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(run_dir)
    print(f"Logging TensorBoard events to: {run_dir}")

    # Checkpoint saving callback function
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    
    def save_checkpoint(current_step, make_policy, params):
        checkpoint_path = os.path.join(checkpoint_directory, f"checkpoint_{current_step}")
        model.save_params(checkpoint_path, params)
        # Write to final_jax_policy so enjoy_jax.py can load it immediately
        
        print(f" -> Checkpoint saved for step {current_step}")

    # Progress callback function
    times = []
    def progress(num_steps, metrics):
        times.append(num_steps)
        print(f"Step {num_steps} - Reward: {metrics['eval/episode_reward']:.2f}")
        for name, value in metrics.items():
            tb_writer.add_scalar(name, float(value), num_steps)

    # Load existing policy parameters for transfer learning / warm restart
    restore_params = None
    if os.path.exists(model_path):
        print(f"Restoring parameters from {model_path} for transfer learning/warm-start...")
        restore_params = model.load_params(model_path)
    else:
        print("No existing policy checkpoint found. Training from scratch...")

    print(f"Training on device: {jax.devices()[0]}")
    print("Starting PPO Training...")
    
    # Run training directly using ppo.train
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=8_000_000,
        num_evals=30,
        reward_scaling=0.02,
        episode_length=400,
        normalize_observations=False,  # Disable normalization as inputs are already scaled in env
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=2,
        discounting=0.99,
        learning_rate=3e-5,
        entropy_cost=2e-3,
        num_envs=4096,
        batch_size=256,
        seed=0,
        progress_fn=progress,
        policy_params_fn=save_checkpoint,
        restore_params=restore_params
    )

    print("Optimization tracking sequence complete.")
    
    # Save model
    model.save_params(output_path, params)
    print(f"Saved optimized JAX policy to {output_path}")

if __name__ == "__main__":
    main()
