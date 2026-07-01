import os
import time
import sys

# Force JAX to allocate memory dynamically as needed instead of pre-claiming 75%
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Limit compile threads to avoid CPU memory spikes
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import jax
from brax import envs
from brax.training.agents.ppo import train as ppo
from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax

def test_oom():
    print("=" * 60)
    # Register env
    try:
        envs.register_environment('krti_gate_jax', KRTIAviaryJax)
    except Exception:
        pass # Already registered

    env = KRTIAviaryJax()

    env_sizes = [16, 32, 64, 128, 512, 1024, 2048, 4096]
    
    for num_envs in env_sizes:
        print("\n" + "-" * 50)
        print(f"Testing environment size: num_envs = {num_envs}")
        print("-" * 50)

        # Tune batch size and minibatches to match environment count
        # batch_size cannot exceed num_envs.
        # num_minibatches must divide batch_size.
        batch_size = min(1024, num_envs)
        if batch_size >= 32:
            num_minibatches = 32
        else:
            num_minibatches = batch_size  # 1 sample per minibatch
            
        print(f"Configured: batch_size={batch_size}, num_minibatches={num_minibatches}")

        # Set up a very short training run to trigger JIT compilation and memory allocations
        try:
            start = time.time()
            make_inference_fn, params, _ = ppo.train(
                environment=env,
                num_timesteps=1000,  # Keep it short to speed up benchmarking
                num_evals=2,
                reward_scaling=0.1,
                episode_length=100,
                normalize_observations=True,
                action_repeat=1,
                unroll_length=10,
                num_minibatches=num_minibatches,
                num_updates_per_batch=2,
                discounting=0.99,
                learning_rate=3e-4,
                entropy_cost=1e-2,
                num_envs=num_envs,
                batch_size=batch_size,
                seed=0,
                progress_fn=lambda steps, metrics: None
            )
            elapsed = time.time() - start
            print(f"SUCCESS: num_envs = {num_envs} ran successfully in {elapsed:.2f} seconds.")
            
        except Exception as e:
            print(f"FAILED: num_envs = {num_envs} failed with error:")
            print(e)
            # If we get a CUDA OOM or JAX ResourceExhaustedError, we print it clearly.
            if "ResourceExhaustedError" in str(e) or "out of memory" in str(e).lower():
                print(">>> IDENTIFIED OUT OF MEMORY (OOM) LIMIT! <<<")
            break

    print("\n" + "=" * 60)
    print("OOM testing complete.")
    print("=" * 60)

if __name__ == "__main__":
    test_oom()
