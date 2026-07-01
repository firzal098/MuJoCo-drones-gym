import os

import jax
import jax.numpy as jnp
from brax import envs
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax


def main():
    # Register the environment
    try:
        envs.register_environment("krti_gate_jax", KRTIAviaryJax)
    except:
        pass
    env = KRTIAviaryJax()

    # Paths
    stage_1_path = "./results/krti_single_rl_jax/finalised/stage_1"
    final_path = "./results/krti_single_rl_jax/finalised/final"

    # We want to compare stage_1 and the new final policy
    for name, model_path in [
        ("Stage 1 Policy", stage_1_path),
        ("Final Policy", final_path),
    ]:
        if not os.path.exists(model_path):
            print(f"{name} not found at {model_path}, skipping.")
            continue

        print(f"\nEvaluating {name}...")
        params = model.load_params(model_path)

        # Build inference network
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size=13,
            action_size=4,
        )
        inference_fn = ppo_networks.make_inference_fn(ppo_network)
        predict_fn = jax.jit(inference_fn(params, deterministic=True))

        # Run parallel evaluation on GPU
        num_envs = 1000
        steps_limit = 400

        # Reset 1000 envs in parallel
        rng = jax.random.PRNGKey(42)
        rngs = jax.random.split(rng, num_envs)

        # Reset env
        reset_fn = jax.jit(jax.vmap(env.reset))
        state = reset_fn(rngs)

        # Define step function for vmap
        step_fn = jax.jit(jax.vmap(env.step))

        # Loop steps
        total_rewards = jnp.zeros(num_envs)
        cleared_gates = jnp.zeros(num_envs)
        crashes = jnp.zeros(num_envs)
        gate_collided = jnp.zeros(num_envs)
        episode_lengths = jnp.zeros(num_envs)
        dones = jnp.zeros(num_envs)

        # We unroll the loop using a python loop over JIT'ed JAX steps for simplicity,
        # but running 400 steps on 1000 parallel environments will be super fast anyway (approx 1-2 seconds)
        for step in range(steps_limit):
            # Split rng for policy
            rng, policy_rng = jax.random.split(rng)
            policy_rngs = jax.random.split(policy_rng, num_envs)

            # Predict actions in parallel
            actions, _ = predict_fn(state.obs, policy_rngs)

            # Step in parallel
            state = step_fn(state, actions)

            # Accumulate metrics for active envs
            active = 1.0 - dones
            total_rewards += state.reward * active
            cleared_gates = jnp.maximum(cleared_gates, state.metrics["cleared_gate"])
            crashes = jnp.maximum(crashes, state.metrics["crashed"])
            gate_collided = jnp.maximum(gate_collided, state.metrics["gate_collided"])
            episode_lengths += active

            # Update dones
            dones = jnp.maximum(dones, state.done)

            # If all are done, break early
            if jnp.all(dones > 0.5):
                break

        # Compute averages
        avg_reward = jnp.mean(total_rewards)
        avg_cleared = jnp.mean(cleared_gates)
        avg_crashed = jnp.mean(crashes)
        avg_collided = jnp.mean(gate_collided)
        avg_length = jnp.mean(episode_lengths)

        print(f"=== {name} Performance (1000 Parallel Episodes) ===")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate (Cleared Gate): {avg_cleared * 100:.2f}%")
        print(f"  Crash Rate: {avg_crashed * 100:.2f}%")
        print(f"  Gate Collision Rate: {avg_collided * 100:.2f}%")
        print(f"  Average Episode Length: {avg_length:.1f} steps")


if __name__ == "__main__":
    main()
