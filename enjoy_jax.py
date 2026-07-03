import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import cv2
import mujoco
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model
import mediapy as media

from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax

# TOGGLES FOR TESTING MODES:
# 1. USE_HEURISTIC_TEST_FLIGHT: Fly towards and through the gate using a proportional controller.
# 2. USE_LOOK_AWAY_TEST_FLIGHT: Spin 180 degrees away from the gate to test if the bounding box cleanly disappears.
# 3. If both are False, the script loads and flies with the trained JAX PPO neural network policy.
# Reset to False by default to prevent overlapping diagnostic overrides.
USE_HEURISTIC_TEST_FLIGHT = False
USE_LOOK_AWAY_TEST_FLIGHT = False

BASE_MODEL_DIR = "./results/krti_single_rl_jax"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained JAX PPO drone policy.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved Brax model params (default: auto-resolved from --curriculum-level)",
    )
    parser.add_argument(
        "--curriculum-level",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Curriculum level for the environment (default: 1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Maximum evaluation steps per episode (default: 600)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run headless batch evaluation (vmap+lax.scan, no rendering, no menu).",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=64,
        help="Number of parallel environments for --evaluate mode (default: 64)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per parallel env for --evaluate mode (default: 10)",
    )
    return parser.parse_args()


def default_model_path(level: int) -> str:
    """Resolve the default checkpoint path for a given curriculum level."""
    return os.path.join(BASE_MODEL_DIR, f"stage_{level}_final")


def load_params_and_network(env, model_path: str):
    """Load checkpoint params and build a PPO network. Returns (params, ppo_network) or (None, None)."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Policy file not found at: {model_path}")
        return None, None

    print(f"Loading JAX policy from: {model_path}")
    params = model.load_params(model_path)

    try:
        checkpoint_obs_size = params[0].mean.shape[0]
    except (AttributeError, IndexError, TypeError):
        checkpoint_obs_size = env.observation_size

    if checkpoint_obs_size != env.observation_size:
        print(f"  WARNING: checkpoint obs_size={checkpoint_obs_size} "
              f"!= env obs_size={env.observation_size}")
        print(f"  Using checkpoint obs_size={checkpoint_obs_size} for network creation.")
    else:
        print(f"  obs_size         : {checkpoint_obs_size} (matches env)")

    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=checkpoint_obs_size,
        action_size=env.action_size,
    )
    return params, ppo_network


def load_predict_fn(env, model_path: str):
    """Load params and build a JIT-compiled inference function for the given env."""
    params, ppo_network = load_params_and_network(env, model_path)
    if params is None:
        return None
    inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return jax.jit(inference_fn(params, deterministic=True))


def run_batch_evaluate(env, model_path: str, num_envs: int, num_episodes: int, episode_length: int):
    """
    Headless, noise-free, GPU-parallelised evaluation.

    Uses:
      - deterministic=True inference (no action noise)
      - jax.vmap  over num_envs parallel environments
      - jax.lax.scan over episode_length timesteps  (single fused GPU kernel)

    Reports mean/std/min/max reward across num_envs * num_episodes roll-outs.
    """
    params, ppo_network = load_params_and_network(env, model_path)
    if params is None:
        return

    # Deterministic inference — strips Gaussian variance, uses dist.mode() internally
    inference_fn = ppo_networks.make_inference_fn(ppo_network)
    raw_predict = inference_fn(params, deterministic=True)

    # Normalisation state lives in params[0]; vmap it across the obs batch
    def batched_predict(obs_batch, rng_batch):
        """Apply policy to [N, obs_dim] observations. rng_batch unused (deterministic)."""
        return jax.vmap(lambda o, r: raw_predict(o, r))(obs_batch, rng_batch)

    # Vectorised env primitives
    vmapped_reset = jax.vmap(env.reset)
    vmapped_step  = jax.vmap(env.step)

    @jax.jit
    def eval_step(carry, _):
        state, rewards, active_mask, rng = carry
        rng, step_rng = jax.random.split(rng)
        rngs = jax.random.split(step_rng, num_envs)
        actions, _ = batched_predict(state.obs, rngs)
        next_state = vmapped_step(state, actions)
        new_rewards = rewards + next_state.reward * active_mask
        new_mask    = active_mask * (1.0 - next_state.done)
        return (next_state, new_rewards, new_mask, rng), None

    @jax.jit
    def run_episode_batch(rng):
        reset_keys  = jax.random.split(rng, num_envs)
        state       = vmapped_reset(reset_keys)
        rewards     = jnp.zeros(num_envs)
        active_mask = jnp.ones(num_envs)
        (_, final_rewards, _, rng_out), _ = jax.lax.scan(
            eval_step, (state, rewards, active_mask, rng), None, length=episode_length
        )
        return final_rewards, rng_out

    print(f"\n{'='*55}")
    print(f"  BATCH EVALUATION  —  Stage {env.curriculum_level}")
    print(f"  Parallel envs  : {num_envs}")
    print(f"  Episodes/env   : {num_episodes}")
    print(f"  Episode length : {episode_length} steps")
    print(f"  Total roll-outs: {num_envs * num_episodes}")
    print(f"  Mode           : deterministic (no action noise)")
    print(f"{'='*55}")
    print("Compiling eval kernel (first episode)...")

    rng = jax.random.PRNGKey(0)
    all_rewards = []

    for ep in range(num_episodes):
        rng, ep_rng = jax.random.split(rng)
        ep_rewards, rng = run_episode_batch(ep_rng)
        ep_rewards_np = np.array(ep_rewards)
        all_rewards.append(ep_rewards_np)
        ep_mean = ep_rewards_np.mean()
        ep_std  = ep_rewards_np.std()
        print(f"  Episode {ep+1:>3}/{num_episodes}  |  mean={ep_mean:8.2f}  std={ep_std:7.2f}  "
              f"min={ep_rewards_np.min():8.2f}  max={ep_rewards_np.max():8.2f}")

    all_rewards_np = np.concatenate(all_rewards)   # shape: [num_envs * num_episodes]
    print(f"\n{'─'*55}")
    print(f"  SUMMARY  ({len(all_rewards_np)} total roll-outs)")
    print(f"  Mean   : {all_rewards_np.mean():.2f}")
    print(f"  Std    : {all_rewards_np.std():.2f}")
    print(f"  Median : {np.median(all_rewards_np):.2f}")
    print(f"  Min    : {all_rewards_np.min():.2f}")
    print(f"  Max    : {all_rewards_np.max():.2f}")
    print(f"{'='*55}\n")


def build_gate_projector(env, cam_id, fovy):
    """Return a closure that computes the gate bounding box for a given mj_data."""
    from multi_drone_mujoco.examples.krti_arena import get_gate_corners, project_point

    def compute_fake_yolo_local(mj_data):
        gate_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")
        if gate_body_id < 0:
            return None

        cam_pos = mj_data.cam_xpos[cam_id].copy()
        cam_mat = mj_data.cam_xmat[cam_id].copy().reshape(3, 3)
        T_gate = mj_data.xpos[gate_body_id].copy()
        R_gate = mj_data.xmat[gate_body_id].copy().reshape(3, 3)

        local_corners = get_gate_corners("single")
        px_list = []

        for pt_local in local_corners:
            pt_world = R_gate @ pt_local + T_gate
            dp = pt_world - cam_pos
            p_cam = cam_mat.T @ dp
            x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]

            if z_c < 0:
                depth = -z_c
                f_y = 1.0 / np.tan(np.deg2rad(fovy) / 2.0)
                f_x = f_y * (240.0 / 320.0)

                ndc_x = f_x * (x_c / depth)
                ndc_y = f_y * (y_c / depth)

                px_x = (ndc_x + 1.0) / 2.0 * 320
                px_y = (1.0 - ndc_y) / 2.0 * 240
                px_list.append((px_x, px_y))

        if not px_list:
            return None

        xs = [p[0] for p in px_list]
        ys = [p[1] for p in px_list]

        if min(xs) > 320 or max(xs) < 0 or min(ys) > 240 or max(ys) < 0:
            return None

        x_min = int(np.clip(min(xs), 0, 320))
        y_min = int(np.clip(min(ys), 0, 240))
        x_max = int(np.clip(max(xs), 0, 320))
        y_max = int(np.clip(max(ys), 0, 240))

        if x_max - x_min < 2 or y_max - y_min < 2:
            return None

        return x_min, y_min, x_max, y_max

    return compute_fake_yolo_local


def run_episode(env, step_fn, predict_fn, max_steps, rng, renderer, mj_data, cam_name, compute_bbox):
    """
    Run a single episode.
    Returns (frames, episode_reward, steps, rng, user_quit)
    where user_quit=True means Esc was pressed mid-episode.
    """
    state = jax.jit(env.reset)(rng)
    frames = []
    episode_reward = 0.0
    steps = 0
    user_quit = False

    print("Starting flight evaluation... Close OpenCV window or press Esc to exit.")
    try:
        while steps < max_steps:
            rng, action_rng = jax.random.split(rng)

            if USE_LOOK_AWAY_TEST_FLIGHT:
                drone_pos = np.array(state.pipeline_state.qpos[0:3])
                gate_pos = np.array(state.info["gate_pos"])
                rel_gate_world = gate_pos - drone_pos

                w, x, y, z = np.array(state.pipeline_state.qpos[3:7])
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)

                rel_gate_body = np.array([
                    rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,
                    -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y),
                    -rel_gate_world[2]
                ])

                act_forward = -0.15
                act_lateral = 0.0
                act_vertical = -float(np.clip(0.7 * rel_gate_body[2], -0.3, 0.3))
                yaw_error_away = np.arctan2(-rel_gate_body[1], -rel_gate_body[0])
                act_yaw = float(np.clip(1.3 * yaw_error_away, -0.4, 0.4))
                action = jnp.array([act_forward, act_lateral, act_vertical, act_yaw])

            elif USE_HEURISTIC_TEST_FLIGHT:
                drone_pos = np.array(state.pipeline_state.qpos[0:3])
                gate_pos = np.array(state.info["gate_pos"])
                rel_gate_world = gate_pos - drone_pos

                w, x, y, z = np.array(state.pipeline_state.qpos[3:7])
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)

                rel_gate_body = np.array([
                    rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,
                    -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y),
                    -rel_gate_world[2]
                ])

                act_forward = float(np.clip(0.35 * rel_gate_body[0], 0.1, 0.55))
                act_lateral = float(np.clip(0.4 * rel_gate_body[1], -0.4, 0.4))
                act_vertical = -float(np.clip(0.7 * rel_gate_body[2], -0.3, 0.3))
                yaw_error = np.arctan2(rel_gate_body[1], rel_gate_body[0])
                act_yaw = float(np.clip(1.3 * yaw_error, -0.4, 0.4))
                action = jnp.array([act_forward, act_lateral, act_vertical, act_yaw])

            else:
                action, _ = predict_fn(state.obs, action_rng)

            state = step_fn(state, action)
            episode_reward += float(state.reward)
            steps += 1

            mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
            mj_data.qvel[:] = np.array(state.pipeline_state.qvel)

            gate_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")
            if gate_body_id >= 0:
                gate_pos = np.array(state.info["gate_pos"])
                gate_body_pos = gate_pos.copy()
                gate_body_pos[0] -= 0.95156
                gate_body_pos[2] = 0.0
                env.mj_model.body_pos[gate_body_id] = gate_body_pos

            mujoco.mj_forward(env.mj_model, mj_data)

            renderer.update_scene(mj_data, camera=cam_name)
            img = renderer.render()

            hud_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cx, cy = 320 // 2, 240 // 2
            cv2.drawMarker(hud_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=1)

            cur_drone_pos = np.array(state.pipeline_state.qpos[0:3])
            cur_gate_pos = np.array(state.info["gate_pos"])
            dist_gate = np.linalg.norm(cur_gate_pos - cur_drone_pos)

            box = compute_bbox(mj_data)
            if box is not None:
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(hud_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                tx, ty = (x_min + x_max) // 2, (y_min + y_max) // 2
                cv2.line(hud_img, (cx, cy), (tx, ty), (0, 0, 255), 1)
                cv2.circle(hud_img, (tx, ty), 4, (0, 0, 255), -1)
                cv2.putText(hud_img, f"LOCK: {steps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(hud_img, "SEARCHING...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.putText(hud_img, f"GATE DIST: {dist_gate:.2f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.imshow("Drone JAX Policy Evaluation", hud_img)
            if cv2.waitKey(30) & 0xFF == 27:  # Esc
                user_quit = True
                break

            frames.append(cv2.cvtColor(hud_img, cv2.COLOR_BGR2RGB))

            if float(state.done) > 0.5:
                print(f"Episode finished! Steps: {steps}, Total Reward: {episode_reward:.2f}")
                break

    except KeyboardInterrupt:
        print("Evaluation stopped.")
        user_quit = True
    finally:
        cv2.destroyAllWindows()

    return frames, episode_reward, steps, rng, user_quit


def prompt_next_action(current_level: int) -> tuple:
    """
    Interactive post-episode menu.
    Returns (action, new_level) where action is one of: 'quit', 'retry', 'stage'.
    """
    print()
    print("=" * 40)
    print(f"  Current stage : {current_level}")
    print("  What next?")
    print("    [q] Quit")
    print("    [r] Retry  (same stage)")
    print("    [s] Switch stage")
    print("=" * 40)

    while True:
        choice = input("  Choice [q/r/s]: ").strip().lower()
        if choice == "q":
            return "quit", current_level
        elif choice == "r":
            return "retry", current_level
        elif choice == "s":
            while True:
                raw = input("  Stage number [1-5]: ").strip()
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    return "stage", int(raw)
                print("  Please enter a number between 1 and 5.")
        else:
            print("  Invalid choice. Enter q, r, or s.")


def main():
    args = parse_args()
    max_steps = args.steps

    # ── Initial setup ─────────────────────────────────────────────────────────
    current_level = args.curriculum_level
    model_path = args.model_path or default_model_path(current_level)

    # ── Batch evaluation mode (headless, no rendering, no menu) ───────────────
    if args.evaluate:
        env = KRTIAviaryJax(curriculum_level=current_level)
        run_batch_evaluate(
            env,
            model_path=model_path,
            num_envs=args.eval_envs,
            num_episodes=args.eval_episodes,
            episode_length=max_steps,
        )
        return

    # Build env + renderer once for the initial stage
    env = KRTIAviaryJax(curriculum_level=current_level)
    renderer = mujoco.Renderer(env.mj_model, height=240, width=320)
    mj_data = mujoco.MjData(env.mj_model)
    cam_name = "drone0_cam"
    cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = env.mj_model.cam_fovy[cam_id]

    print(f"  curriculum_level : {current_level}")
    print(f"  env obs_size     : {env.observation_size}")
    print(f"  max_steps        : {max_steps}")

    # JIT compile step fn once — reused across all retries of the same stage
    step_fn = jax.jit(env.step)

    # Load initial policy
    predict_fn = None
    if not USE_HEURISTIC_TEST_FLIGHT and not USE_LOOK_AWAY_TEST_FLIGHT:
        predict_fn = load_predict_fn(env, model_path)
        if predict_fn is None:
            return
    elif USE_LOOK_AWAY_TEST_FLIGHT:
        print("[LOOK-AWAY MODE] Turning nose 180 degrees away from gate to verify bounding box occlusion.")
    else:
        print("[HEURISTIC MODE] Flying via closed-loop Proportional Controller to verify gate bounding box math.")

    # Gate bbox projector (closed over current env/cam)
    compute_bbox = build_gate_projector(env, cam_id, fovy)

    rng = jax.random.PRNGKey(42)
    ep_count = 0

    # ══ Main evaluation loop ══════════════════════════════════════════════════
    while True:
        ep_count += 1
        print(f"\nResetting environment... [Stage {current_level} | Episode {ep_count}]")

        frames, episode_reward, steps, rng, user_quit = run_episode(
            env, step_fn, predict_fn, max_steps,
            rng, renderer, mj_data, cam_name, compute_bbox
        )

        # Save video with stage suffix
        if frames:
            video_path = f"/home/firza/MuJoCo-drones-gym/enjoy_jax_flight_stage{current_level}.mp4"
            print(f"Saving flight video to {video_path}...")
            media.write_video(video_path, frames, fps=24)
            print(f"Video saved to {video_path}!")

        # ── Post-episode menu ─────────────────────────────────────────────────
        action, new_level = prompt_next_action(current_level)

        if action == "quit":
            print("Goodbye!")
            break

        elif action == "retry":
            # Same env, same step_fn, same predict_fn — just loop again
            continue

        elif action == "stage":
            if new_level == current_level:
                print(f"  Already on stage {current_level}, retrying...")
                continue

            print(f"\n  Switching to stage {new_level}...")
            current_level = new_level

            # Rebuild env for the new curriculum level
            env = KRTIAviaryJax(curriculum_level=current_level)
            renderer = mujoco.Renderer(env.mj_model, height=240, width=320)
            mj_data = mujoco.MjData(env.mj_model)
            cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            fovy = env.mj_model.cam_fovy[cam_id]

            # Re-JIT step fn for the new env (unavoidable when env object changes)
            print("  Re-compiling step function for new env...")
            step_fn = jax.jit(env.step)

            # Swap in the checkpoint for the new stage
            if not USE_HEURISTIC_TEST_FLIGHT and not USE_LOOK_AWAY_TEST_FLIGHT:
                new_model_path = default_model_path(new_level)
                new_predict_fn = load_predict_fn(env, new_model_path)
                if new_predict_fn is not None:
                    predict_fn = new_predict_fn
                else:
                    print(f"  [WARN] No checkpoint found for stage {new_level}, keeping previous policy.")

            # Rebuild bbox projector for new env/cam
            compute_bbox = build_gate_projector(env, cam_id, fovy)

            ep_count = 0  # Reset episode counter for new stage
            continue


if __name__ == "__main__":
    main()
