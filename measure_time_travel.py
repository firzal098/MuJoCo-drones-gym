"""
Measure real travel time to gate using a jitted proportional heuristic
controller (same FRD body-frame math as enjoy_jax_flight.py), across many
reset seeds to cover the full spawn distance range.

Run in your repo root:
    python -u measure_travel_time_v2.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax


def heuristic_action(state):
    drone_pos = state.pipeline_state.qpos[0:3]
    gate_pos = state.info["gate_pos"]
    rel_gate_world = gate_pos - drone_pos

    w, x, y, z = state.pipeline_state.qpos[3:7]
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)

    rel_gate_body = jnp.array([
        rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,
        -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y),
        -rel_gate_world[2],
    ])

    act_forward = jnp.clip(0.35 * rel_gate_body[0], 0.1, 0.55)
    act_lateral = jnp.clip(0.4 * rel_gate_body[1], -0.4, 0.4)
    act_vertical = jnp.clip(0.7 * rel_gate_body[2], -0.3, 0.3)
    yaw_error = jnp.arctan2(rel_gate_body[1], rel_gate_body[0])
    act_yaw = jnp.clip(1.3 * yaw_error, -0.4, 0.4)

    return jnp.array([act_forward, act_lateral, act_vertical, act_yaw])


def run_episode(env, reset_fn, step_fn, rng, max_steps=600):
    state = reset_fn(rng)
    steps = 0
    cleared = False
    crashed = False

    for i in range(max_steps):
        action = heuristic_action(state)
        state = step_fn(state, action)
        steps += 1

        if float(state.metrics["cleared_gate"]) > 0.5:
            cleared = True
            break
        if float(state.metrics["crashed"]) > 0.5:
            crashed = True
            break
        if float(state.done) > 0.5:
            break

    return steps, cleared, crashed


def main():
    print("Building env...")
    env = KRTIAviaryJax()

    print("Jitting reset/step (first call will take a bit to compile)...")
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    results = []
    for trial in range(30):
        rng = jax.random.PRNGKey(trial)
        steps, cleared, crashed = run_episode(env, reset_fn, step_fn, rng)
        results.append((trial, steps, cleared, crashed))
        print(f"trial {trial:2d} | steps {steps:4d} | cleared={cleared} | crashed={crashed}", flush=True)

    steps_arr = np.array([r[1] for r in results])
    cleared_arr = np.array([r[2] for r in results])

    print("\n--- summary ---")
    print(f"clear rate (heuristic P-controller): {cleared_arr.mean()*100:.1f}%")
    if cleared_arr.any():
        clear_steps = steps_arr[cleared_arr]
        print(f"steps-to-clear (successful trials only): "
              f"min={clear_steps.min()} mean={clear_steps.mean():.1f} max={clear_steps.max()}")
    print(f"all trials: min={steps_arr.min()} mean={steps_arr.mean():.1f} max={steps_arr.max()}")
    print("\nif max clear-steps is close to or above your episode_length,")
    print("that length is too tight for the far end of your spawn distribution.")


if __name__ == "__main__":
    main()