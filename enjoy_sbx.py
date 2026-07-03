"""
enjoy_sbx.py
============
Evaluation / visualisation script for trained SBX PPO policies.

Works with models saved by EITHER:
  - train_single_rl_sbx.py     (100-dim obs, standard MuJoCo CPU env)
  - train_single_rl_sbx_mjx.py (20-dim obs, MJX GPU env)

The correct env / obs-dim is auto-detected from the loaded zip file.
Renders a live FPV HUD via OpenCV and saves the flight as an MP4.

Usage:
    # Auto-detect latest stage final from sbx run:
    python enjoy_sbx.py

    # Explicit path (no .zip extension):
    python enjoy_sbx.py --model results/krti_single_rl_sbx/stage_1_final_sbx_brain

    # MJX model:
    python enjoy_sbx.py --model results/krti_single_rl_sbx_mjx/stage_1_final_sbx_mjx_brain

    # Specific checkpoint:
    python enjoy_sbx.py --model results/krti_single_rl_sbx/checkpoints/sbx_stage1_brain_500000_steps

    # Heuristic mode (no model needed):
    python enjoy_sbx.py --heuristic
"""

import os
import argparse
import glob
import numpy as np
import cv2
import mujoco
import jax
import jax.numpy as jnp
import mediapy as media

# ── SBX ───────────────────────────────────────────────────────────────────────
try:
    from sbx import PPO
except ImportError as e:
    raise ImportError(
        "SBX is not installed. Run:\n  pip install sbx-rl"
    ) from e

# ── MJX environment (used for rendering regardless of training backend) ───────
from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax
from multi_drone_mujoco.examples.krti_arena import get_gate_corners, project_point

MAX_STEPS     = 600
VIDEO_FPS     = 24
IMG_W, IMG_H  = 320, 240


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained SBX PPO drone policy.")
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to saved model (without .zip). Auto-detected if omitted."
    )
    p.add_argument(
        "--heuristic", action="store_true",
        help="Run proportional heuristic controller instead of loading a model."
    )
    p.add_argument(
        "--curriculum-level", type=int, default=1,
        help="KRTIAviaryJax curriculum level (1-5) for domain randomisation (default: 1)."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for episode reset (default: 42)."
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Skip saving the flight MP4."
    )
    p.add_argument(
        "--video-path", type=str, default="./enjoy_sbx_flight.mp4",
        help="Output video path (default: ./enjoy_sbx_flight.mp4)."
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model auto-detection
# ══════════════════════════════════════════════════════════════════════════════
def auto_detect_model():
    """Return path to the most recent stage-final SBX model (no .zip)."""
    candidates = []
    for pattern in [
        "results/krti_single_rl_sbx_mjx/stage_*_final_sbx_mjx_brain.zip",
        "results/krti_single_rl_sbx/stage_*_final_sbx_brain.zip",
        "results/krti_single_rl_sbx_mjx/final_krti_sbx_mjx_brain.zip",
        "results/krti_single_rl_sbx/final_krti_sbx_brain.zip",
        "results/krti_single_rl_sbx/checkpoints/sbx_stage*_brain_*_steps.zip",
        "results/krti_single_rl_sbx_mjx/checkpoints/sbx_mjx_stage*_brain_*_steps.zip",
    ]:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    # Pick the most recently modified file
    latest = max(candidates, key=os.path.getmtime)
    return latest[:-4]   # strip .zip


# ══════════════════════════════════════════════════════════════════════════════
# YOLO bounding-box computation (CPU / numpy, matches enjoy_jax.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
def compute_yolo_box(mj_data, mj_model, cam_id, fovy):
    gate_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")
    if gate_body_id < 0:
        return None

    cam_pos = mj_data.cam_xpos[cam_id].copy()
    cam_mat = mj_data.cam_xmat[cam_id].copy().reshape(3, 3)
    T_gate  = mj_data.xpos[gate_body_id].copy()
    R_gate  = mj_data.xmat[gate_body_id].copy().reshape(3, 3)

    local_corners = get_gate_corners("single")
    px_list = []
    for pt_local in local_corners:
        pt_world = R_gate @ pt_local + T_gate
        dp = pt_world - cam_pos
        p_cam = cam_mat.T @ dp
        x_c, y_c, z_c = p_cam
        if z_c < 0:
            depth = -z_c
            f_y = 1.0 / np.tan(np.deg2rad(fovy) / 2.0)
            f_x = f_y * (IMG_H / IMG_W)
            ndc_x = f_x * (x_c / depth)
            ndc_y = f_y * (y_c / depth)
            px_x = (ndc_x + 1.0) / 2.0 * IMG_W
            px_y = (1.0 - ndc_y) / 2.0 * IMG_H
            px_list.append((px_x, px_y))

    if not px_list:
        return None

    xs = [p[0] for p in px_list]
    ys = [p[1] for p in px_list]

    if min(xs) > IMG_W or max(xs) < 0 or min(ys) > IMG_H or max(ys) < 0:
        return None

    x_min = int(np.clip(min(xs), 0, IMG_W))
    y_min = int(np.clip(min(ys), 0, IMG_H))
    x_max = int(np.clip(max(xs), 0, IMG_W))
    y_max = int(np.clip(max(ys), 0, IMG_H))

    if x_max - x_min < 2 or y_max - y_min < 2:
        return None

    return x_min, y_min, x_max, y_max


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic proportional controller (same as enjoy_jax.py)
# ══════════════════════════════════════════════════════════════════════════════
def heuristic_action(state):
    """Proportional controller flying towards the gate."""
    drone_pos = np.array(state.pipeline_state.qpos[0:3])
    gate_pos  = np.array(state.info["gate_pos"])
    rel_world = gate_pos - drone_pos

    w, x, y, z = np.array(state.pipeline_state.qpos[3:7])
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cy, sy = np.cos(yaw), np.sin(yaw)

    rel_body = np.array([
        rel_world[0] * cy + rel_world[1] * sy,
        -(-rel_world[0] * sy + rel_world[1] * cy),
        -rel_world[2],
    ])

    act_fwd  = float(np.clip(0.35 * rel_body[0], 0.1, 0.55))
    act_lat  = float(np.clip(0.40 * rel_body[1], -0.4, 0.4))
    act_vert = -float(np.clip(0.70 * rel_body[2], -0.3, 0.3))
    yaw_err  = np.arctan2(rel_body[1], rel_body[0])
    act_yaw  = float(np.clip(1.30 * yaw_err,  -0.4, 0.4))

    return np.array([act_fwd, act_lat, act_vert, act_yaw], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Resolve model path ────────────────────────────────────────────────────
    model_path = None
    if not args.heuristic:
        model_path = args.model or auto_detect_model()
        if model_path is None:
            print(
                "[ERROR] No saved SBX model found. Train first with:\n"
                "  python train_single_rl_sbx.py\n"
                "  python train_single_rl_sbx_mjx.py\n"
                "Or run with --heuristic for a P-controller baseline."
            )
            return
        if not os.path.exists(model_path + ".zip"):
            print(f"[ERROR] Model file not found: {model_path}.zip")
            return

    # ── Build MJX environment ─────────────────────────────────────────────────
    print(f"[ENV] Building KRTIAviaryJax (curriculum_level={args.curriculum_level}) ...")
    env = KRTIAviaryJax(curriculum_level=args.curriculum_level)

    # ── Load SBX PPO model ────────────────────────────────────────────────────
    policy = None
    if not args.heuristic:
        print(f"[MODEL] Loading SBX PPO from: {model_path}.zip")
        policy = PPO.load(model_path)
        obs_dim = policy.observation_space.shape[0]
        print(f"  Policy obs dim : {obs_dim}")
        print(f"  Env obs dim    : {env.observation_size}")
        if obs_dim != env.observation_size:
            print(
                f"  [WARN] Obs dim mismatch! Model expects {obs_dim}-dim but env "
                f"produces {env.observation_size}-dim.\n"
                "  This model was likely trained with train_single_rl_sbx.py (100-dim).\n"
                "  Use a model from train_single_rl_sbx_mjx.py (20-dim) for this env.\n"
                "  Continuing anyway — policy output may be invalid."
            )
    else:
        print("[MODE] Heuristic proportional controller (no model).")

    # ── MuJoCo CPU renderer setup ─────────────────────────────────────────────
    renderer = mujoco.Renderer(env.mj_model, height=IMG_H, width=IMG_W)
    mj_data  = mujoco.MjData(env.mj_model)
    cam_name = "drone0_cam"
    cam_id   = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy     = env.mj_model.cam_fovy[cam_id]
    gate_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")

    # ── Reset environment ─────────────────────────────────────────────────────
    print(f"[RESET] seed={args.seed} ...")
    rng   = jax.random.PRNGKey(args.seed)
    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)
    state = reset_fn(rng)

    # ── Flight loop ───────────────────────────────────────────────────────────
    frames         = []
    episode_reward = 0.0
    steps          = 0

    print(f"Starting evaluation (max {MAX_STEPS} steps). Press Esc to quit.")
    try:
        while steps < MAX_STEPS:
            # ── Compute action ────────────────────────────────────────────────
            if args.heuristic:
                action_np = heuristic_action(state)
            else:
                obs_np    = np.array(state.obs, dtype=np.float32)[np.newaxis]   # (1, obs_dim)
                action_np, _ = policy.predict(obs_np, deterministic=True)
                action_np = action_np[0]                                          # (act_dim,)

            # ── Step MJX physics ──────────────────────────────────────────────
            action_jax = jnp.array(action_np, dtype=jnp.float32)
            state      = step_fn(state, action_jax)
            episode_reward += float(state.reward)
            steps += 1

            # ── Sync JAX state → MuJoCo CPU (for rendering only) ─────────────
            mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
            mj_data.qvel[:] = np.array(state.pipeline_state.qvel)

            # Sync randomised gate position to the CPU model
            if gate_body_id >= 0:
                gate_pos_jax = np.array(state.info["gate_pos"])
                body_pos     = gate_pos_jax.copy()
                body_pos[0] -= 0.95156   # gate body offset (same as enjoy_jax.py)
                body_pos[2]  = 0.0
                env.mj_model.body_pos[gate_body_id] = body_pos

            mujoco.mj_forward(env.mj_model, mj_data)

            # ── Render FPV frame ──────────────────────────────────────────────
            renderer.update_scene(mj_data, camera=cam_name)
            img     = renderer.render()
            hud     = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cx, cy  = IMG_W // 2, IMG_H // 2
            cv2.drawMarker(hud, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)

            # Gate distance
            cur_pos   = np.array(state.pipeline_state.qpos[0:3])
            gate_pos  = np.array(state.info["gate_pos"])
            dist_gate = np.linalg.norm(gate_pos - cur_pos)

            # YOLO bounding box overlay
            box = compute_yolo_box(mj_data, env.mj_model, cam_id, fovy)
            if box is not None:
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                tx, ty = (x_min + x_max) // 2, (y_min + y_max) // 2
                cv2.line(hud, (cx, cy), (tx, ty), (0, 0, 255), 1)
                cv2.circle(hud, (tx, ty), 4, (0, 0, 255), -1)
                cv2.putText(hud, f"LOCK  step:{steps}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(hud, "SEARCHING...", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.putText(hud, f"DIST: {dist_gate:.2f}m", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(hud, f"R: {episode_reward:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Mode label
            label = "HEURISTIC" if args.heuristic else "SBX PPO"
            cv2.putText(hud, label, (IMG_W - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            cv2.imshow("SBX Policy — Drone FPV", hud)
            if cv2.waitKey(30) & 0xFF == 27:   # Esc to quit
                break

            frames.append(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))

            # ── Episode end ───────────────────────────────────────────────────
            if float(state.done) > 0.5:
                cleared = float(state.metrics.get("cleared_gate", 0.0)) > 0.5
                crashed = float(state.metrics.get("crashed",       0.0)) > 0.5
                status  = "GATE CLEARED ✓" if cleared else ("CRASHED ✗" if crashed else "TIMEOUT")
                print(f"\n[DONE] {status}  |  Steps: {steps}  |  Total reward: {episode_reward:.2f}")
                break

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
    finally:
        cv2.destroyAllWindows()

    print(f"\nFlight summary — Steps: {steps} | Total reward: {episode_reward:.2f}")

    # ── Save video ────────────────────────────────────────────────────────────
    if frames and not args.no_video:
        print(f"Saving flight video → {args.video_path} ...")
        media.write_video(args.video_path, frames, fps=VIDEO_FPS)
        print(f"Video saved: {args.video_path}")


if __name__ == "__main__":
    main()
