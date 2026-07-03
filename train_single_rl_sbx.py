"""
train_single_rl_sbx.py
======================
Single-gate drone navigation training using Stable-Baselines Jax (SBX).

This script combines:
  - The full MuJoCo simulation wrapper (SingleGateTrainingWrapper) from
    train_single_rl_v2.py, including:
      * Simulated YOLO bounding-box observations
      * GuidedVehicle ArduPilot-style velocity controller
      * Per-episode domain randomisation (gate + drone spawn)
      * Rich reward shaping (progress, heading, height, YOLO alignment,
        survival, gate-clear bonus, crash penalties)
  - The 5-stage curriculum scheduler from train_single_rl_jax.py
  - Stable-Baselines Jax (SBX) PPO as the RL backend (JAX-accelerated,
    SB3-compatible API)

Install SBX before running:
    pip install sbx-rl
    # or the latest dev build:
    pip install git+https://github.com/araffin/sbx
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. Environment variables — must be set BEFORE importing jax / jaxlib
# ──────────────────────────────────────────────────────────────────────────────
import os

# ── JAX / XLA GPU settings ────────────────────────────────────────────────────
# Allocate JAX device memory on-demand instead of pre-claiming 75%
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# Limit compiler threads to reduce peak host-RAM spikes during JIT
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=1")
# Cap JAX VRAM usage at 70% — leaves headroom for MuJoCo (CPU) and OS
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")
# Force JAX to use GPU if available (set to 'cpu' to disable GPU)
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ──────────────────────────────────────────────────────────────────────────────
# 1. JAX compilation cache (warm restarts are significantly faster)
# ──────────────────────────────────────────────────────────────────────────────
import jax

_jax_cache = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"),
)
os.makedirs(_jax_cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache)

# ──────────────────────────────────────────────────────────────────────────────
# 2. SBX imports (Stable-Baselines Jax)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sbx import PPO
except ImportError as e:
    raise ImportError(
        "SBX is not installed. Please run:\n"
        "  pip install sbx-rl\n"
        "or:\n"
        "  pip install git+https://github.com/araffin/sbx"
    ) from e

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

# ──────────────────────────────────────────────────────────────────────────────
# 3. Custom aviary / arena components (from train_single_rl_v2.py)
# ──────────────────────────────────────────────────────────────────────────────
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType
from multi_drone_mujoco.examples.krti_arena import (
    KRTIAviary,
    get_gate_corners,
    project_point,
)
from multi_drone_mujoco.control.guided_mode import GuidedVehicle

# ──────────────────────────────────────────────────────────────────────────────
# 4. Global flags
# ──────────────────────────────────────────────────────────────────────────────
DEBUG_GUI    = False
RECORD_GIF   = False   # Set True to record episodes and save as GIF
SHOW_FPV_GUI = False   # Set True to spawn a custom FPV target-tracker window

# Number of parallel environment workers per curriculum stage
NUM_ENVS = 1 if (DEBUG_GUI or SHOW_FPV_GUI) else 16


# ══════════════════════════════════════════════════════════════════════════════
# 5. Environment wrapper  (unchanged from train_single_rl_v2.py)
#    Policies from v2 are preserved exactly — observation / action spaces and
#    reward shaping are identical so that saved checkpoints stay compatible.
# ══════════════════════════════════════════════════════════════════════════════
class SingleGateTrainingWrapper(gym.Wrapper):
    """
    Custom wrapper targeting gate_single_a of the KRTI 2026 arena.

    Includes:
      * Domain randomisation (gate position + drone spawn)
      * Simulated YOLO bounding-box observations (no pixel rendering)
      * ArduPilot GUIDED-mode velocity controller
      * 100-dim observation space matching train_single_rl_v2 policies
    """

    def __init__(self, env, rank: int = 0):
        super().__init__(env)
        self.rank = rank

        # Ground-truth nominal gate position
        self.nominal_gate_pos = np.array([0.17, 12.26, 1.0])
        self.gate_targets = [
            {"name": "gate_single_a", "pos": self.nominal_gate_pos.copy(), "type": "single"}
        ]
        self.max_episode_steps = 600
        self.current_step      = 0

        # Noisy gate-position estimate (overwritten at each reset)
        self.noisy_gate_position = self.nominal_gate_pos.copy()

        # Action space: [V_fwd, V_lat, V_down, Yaw_Rate] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # ArduPilot GUIDED-mode velocity controller
        self.vehicle = GuidedVehicle(self.env, drone_index=0)

        # GIF / FPV recording
        self.record_gif   = RECORD_GIF
        self.gif_frames   = []
        self._gif_renderer = None
        self.show_fpv_gui  = SHOW_FPV_GUI
        self._fpv_renderer = None

        # Episode-level metric accumulators (reset every episode)
        self._ep_reward_progress  = 0.0
        self._ep_reward_heading   = 0.0
        self._ep_reward_height    = 0.0
        self._ep_reward_alignment = 0.0
        self._ep_reward_action    = 0.0
        self._ep_reward_time      = 0.0
        self._ep_reward_terminal  = 0.0
        self._ep_cleared_gate     = 0
        self._ep_gate_collided    = 0
        self._ep_crashed          = 0

        # Observation space: 100-dim (identical to v2 policies)
        # [YOLO_Box(4), Rel_Noisy_Gate(3), EKF_Vel(3), Attitude(3), Padding(87)]
        self.observation_space = spaces.Box(
            low=-1.2, high=1.2, shape=(100,), dtype=np.float32
        )

        # Camera parameters
        self.cam_name = "drone0_cam"
        self.cam_id   = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name
        )
        self.img_w, self.img_h = 320, 240
        self.fovy = self.env.model.cam_fovy[self.cam_id]

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_fake_yolo(self):
        """Calculates 2D screen bounding-boxes mathematically (no pixel render)."""
        gate      = self.gate_targets[0]
        body_id   = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, gate["name"]
        )

        if body_id < 0:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        cam_pos = self.env.data.cam_xpos[self.cam_id].copy()
        cam_mat = self.env.data.cam_xmat[self.cam_id].copy()
        T_gate  = self.env.data.xpos[body_id].copy()
        R_gate  = self.env.data.xmat[body_id].copy().reshape(3, 3)

        local_corners = get_gate_corners(gate["type"])
        px_list = []

        for pt_local in local_corners:
            pt_world = R_gate @ pt_local + T_gate
            px = project_point(
                pt_world, cam_pos, cam_mat, self.fovy, self.img_w, self.img_h
            )
            if px is not None:
                px_x, px_y = px
                if 0 <= px_x <= self.img_w and 0 <= px_y <= self.img_h:
                    px_list.append(px)

        if len(px_list) < 2:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        xs    = [p[0] for p in px_list]
        ys    = [p[1] for p in px_list]
        x_min = np.clip(min(xs) / self.img_w, 0.0, 1.0)
        x_max = np.clip(max(xs) / self.img_w, 0.0, 1.0)
        y_min = np.clip(min(ys) / self.img_h, 0.0, 1.0)
        y_max = np.clip(max(ys) / self.img_h, 0.0, 1.0)

        # Vision tracking jitter
        yolo_noise = np.random.normal(0, 0.015, size=4)
        noisy_box  = np.array([x_min, y_min, x_max, y_max]) + yolo_noise

        # 5% chance of visual frame dropout
        if np.random.rand() < 0.05:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        return np.clip(noisy_box, 0.0, 1.0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _get_obs(self):
        drone_pos               = self.env.pos[0]
        vx_world, vy_world, vz_world = self.env.vel[0]
        roll, pitch, yaw        = self.env.rpy[0]

        # Relative gate vector in drone body frame
        rel_gate_world = self.noisy_gate_position - drone_pos
        cos_y, sin_y   = np.cos(yaw), np.sin(yaw)
        rel_gate_body  = np.array([
             rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,    # Forward
            -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y), # Right (FRD)
            -rel_gate_world[2],                                          # Down  (FRD)
        ])

        # EKF velocity in body frame
        vel_body = np.array([
             vx_world * cos_y + vy_world * sin_y,
            -(-vx_world * sin_y + vy_world * cos_y),
            -vz_world,
        ])

        # Simulated YOLO bounding-box [err_x, err_y, box_w, box_h]
        yolo_raw = self._compute_fake_yolo()
        if yolo_raw[0] >= 0.0:
            x_min, y_min, x_max, y_max = yolo_raw
            err_x  = ((x_min + x_max) / 2.0) - 0.5
            err_y  = ((y_min + y_max) / 2.0) - 0.5
            box_w  = x_max - x_min
            box_h  = y_max - y_min
            yolo_obs = np.array([err_x, err_y, box_w, box_h], dtype=np.float32)
        else:
            yolo_obs = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        yaw_error = np.arctan2(rel_gate_body[1], rel_gate_body[0])

        rel_gate_body_scaled = np.clip(rel_gate_body / 20.0, -1.0, 1.0)
        vel_body_scaled      = np.clip(vel_body / 4.0, -1.0, 1.0)
        pitch_frd            = -pitch  # MuJoCo nose-down -> FRD nose-up
        attitude_scaled      = np.clip(
            np.array([roll, pitch_frd, yaw_error]) / np.pi, -1.0, 1.0
        )
        padding = np.zeros(87, dtype=np.float32)

        return np.concatenate([
            yolo_obs,
            rel_gate_body_scaled,
            vel_body_scaled,
            attitude_scaled,
            padding,
        ]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        # Domain Randomisation — Gate position
        base_gate_y = np.random.uniform(8.0, 18.0)
        self.nominal_gate_pos = np.array([0.17, base_gate_y, 1.0])
        gx = np.random.uniform(-1.0, 1.0)
        gy = np.random.uniform(-1.0, 1.0)

        body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a"
        )
        if body_id >= 0:
            self.env.model.body_pos[body_id] = np.array([0.17 + gx, base_gate_y + gy, 0.0])

        self.gate_targets[0]["pos"] = self.nominal_gate_pos + np.array([gx, gy, 0.0])

        # Domain Randomisation — Drone spawn
        dx = np.random.uniform(-1.0, 1.0)
        dy = np.random.uniform(-1.0, 1.0)
        self.env.INIT_XYZS[0] = np.array([0.92 + dx, 24.47 + dy, 0.25])

        # Domain Randomisation — Spawn heading
        dyaw = np.random.uniform(-0.3, 0.3)
        self.env.INIT_RPYS[0] = np.array([0.0, 0.0, -np.pi / 2 + dyaw])

        obs, info = self.env.reset(seed=seed, options=options)

        if self.record_gif:
            self.gif_frames = []
            self.gif_frames.append(self.capture_frame())

        # Arm, take-off, stabilise
        self.vehicle.disarm()
        self.vehicle.arm()
        self.vehicle.simple_takeoff(1.0)

        for _ in range(150):
            if self.vehicle.mode == "GUIDED":
                break
            rpm = self.vehicle.update(self.env.CTRL_TIMESTEP)
            self.env.step(rpm)
            if self.env.render_mode == "human":
                self.env.render()
            if self.show_fpv_gui:
                self.render_fpv_hud()
            if self.record_gif:
                self.gif_frames.append(self.capture_frame())

        for _ in range(20):
            rpm = self.vehicle.update(self.env.CTRL_TIMESTEP)
            self.env.step(rpm)
            if self.env.render_mode == "human":
                self.env.render()
            if self.show_fpv_gui:
                self.render_fpv_hud()
            if self.record_gif:
                self.gif_frames.append(self.capture_frame())

        self.active_gate_idx = 0
        self.current_step    = 0

        # Reset episode metric accumulators
        self._ep_reward_progress  = 0.0
        self._ep_reward_heading   = 0.0
        self._ep_reward_height    = 0.0
        self._ep_reward_alignment = 0.0
        self._ep_reward_action    = 0.0
        self._ep_reward_time      = 0.0
        self._ep_reward_terminal  = 0.0
        self._ep_cleared_gate     = 0
        self._ep_gate_collided    = 0
        self._ep_crashed          = 0

        # Noisy gate-position estimate (simulates RTK-GPS / SLAM map accuracy)
        map_noise    = np.random.normal(0, 0.20, size=3)
        map_noise[2] = 0.0
        self.noisy_gate_position = self.gate_targets[0]["pos"] + map_noise

        return self._get_obs(), info

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action):
        self.current_step += 1

        max_xy_speed  = 3.0   # m/s
        max_z_speed   = 1.5   # m/s
        max_yaw_rate  = 1.8   # rad/s

        # Map NN FRD output -> GuidedVehicle FLU
        vx_body  =  action[0] * max_xy_speed
        vy_body  = -action[1] * max_xy_speed
        vz_body  = -action[2] * max_z_speed
        yaw_rate = -action[3] * max_yaw_rate

        self.vehicle.set_velocity(vx_body, vy_body, vz_body, yaw_rate)
        rpm = self.vehicle.update(self.env.CTRL_TIMESTEP)

        target_gate = self.gate_targets[0]["pos"]
        dist_before = np.linalg.norm(target_gate - self.env.pos[0])

        _, _, _, _, info = self.env.step(rpm)

        if self.env.render_mode == "human":
            self.env.render()
        if self.show_fpv_gui:
            self.render_fpv_hud()

        dist_after = np.linalg.norm(target_gate - self.env.pos[0])

        # ── Reward components (tracked individually for TensorBoard) ─────────
        # 1. Trajectory progression
        r_progress = (dist_before - dist_after) * 20.0

        # 2. Heading alignment penalty
        drone_pos      = self.env.pos[0]
        _, _, yaw      = self.env.rpy[0]
        rel_gate_world = self.noisy_gate_position - drone_pos
        cos_y, sin_y   = np.cos(yaw), np.sin(yaw)
        rel_gate_bx    =  rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y
        rel_gate_by    = -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y)
        yaw_error      = np.arctan2(rel_gate_by, rel_gate_bx)
        r_heading      = -0.15 * abs(yaw_error)

        # 3. Vertical alignment penalty
        height_error = -rel_gate_world[2]
        r_height     = -0.15 * abs(height_error)

        # 4. YOLO alignment penalty
        yolo_box = self._compute_fake_yolo()
        if yolo_box[0] >= 0.0:
            x_center    = (yolo_box[0] + yolo_box[2]) / 2.0
            y_center    = (yolo_box[1] + yolo_box[3]) / 2.0
            err_x       = x_center - 0.5
            err_y       = y_center - 0.5
            r_alignment = -(1.0 * np.sqrt(err_x ** 2 + err_y ** 2))
        else:
            r_alignment = -0.20  # Penalise loss of visual contact

        terminated = False
        truncated  = False
        r_terminal = 0.0

        # 5. Successful gate passage
        if dist_after < 0.25:
            center_accuracy_bonus = (0.25 - dist_after) * 800.0
            r_terminal           += 200.0 + center_accuracy_bonus
            self._ep_cleared_gate = 1
            terminated            = True

        # 6. Gate-frame collision
        if self._check_gate_collision():
            r_terminal           -= 450.0
            self._ep_gate_collided = 1
            terminated             = True

        # 7. Safety constraints
        rpy         = self.env.rpy[0]
        has_crashed = (
            self.env.pos[0, 2] < 0.08
            or self.env.pos[0, 2] > 4.5
            or abs(rpy[0]) > np.pi / 2.5
            or abs(rpy[1]) > np.pi / 2.5
        )
        if has_crashed:
            r_terminal      -= 200.0
            self._ep_crashed = 1
            terminated       = True

        # 8. Action smoothness penalty
        r_action = -0.05 * float(np.sum(np.square(action)))

        # 9. Time penalty
        r_time = -0.10

        # Accumulate into episode buckets
        self._ep_reward_progress  += r_progress
        self._ep_reward_heading   += r_heading
        self._ep_reward_height    += r_height
        self._ep_reward_alignment += r_alignment
        self._ep_reward_action    += r_action
        self._ep_reward_time      += r_time
        self._ep_reward_terminal  += r_terminal

        reward = (r_progress + r_heading + r_height + r_alignment
                  + r_terminal + r_action + r_time)

        if self.current_step >= self.max_episode_steps:
            truncated = True

        # Inject episode summary into info at the end of each episode
        if terminated or truncated:
            info["ep_metrics"] = {
                "reward/progress":        self._ep_reward_progress,
                "reward/heading_penalty": self._ep_reward_heading,
                "reward/height_penalty":  self._ep_reward_height,
                "reward/yolo_alignment":  self._ep_reward_alignment,
                "reward/action_penalty":  self._ep_reward_action,
                "reward/time_penalty":    self._ep_reward_time,
                "reward/terminal":        self._ep_reward_terminal,
                "events/cleared_gate":    float(self._ep_cleared_gate),
                "events/gate_collided":   float(self._ep_gate_collided),
                "events/crashed":         float(self._ep_crashed),
                "env/gate_distance_final": float(dist_after),
            }

        if self.record_gif:
            self.gif_frames.append(self.capture_frame())
        if (terminated or truncated) and self.record_gif:
            self.save_gif("./results/trajectory.gif")

        return self._get_obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    def _check_gate_collision(self):
        """Returns True if the drone has collided with any gate geometry."""
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            body1_id = self.env.model.geom_bodyid[contact.geom1]
            body2_id = self.env.model.geom_bodyid[contact.geom2]
            name1 = mujoco.mj_id2name(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, body1_id
            )
            name2 = mujoco.mj_id2name(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, body2_id
            )
            is_drone = (
                (name1 is not None and "drone0" in name1)
                or (name2 is not None and "drone0" in name2)
            )
            is_gate = (
                (name1 is not None and "gate_" in name1)
                or (name2 is not None and "gate_" in name2)
            )
            if is_drone and is_gate:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    def capture_frame(self):
        """Captures a third-person tracking frame for GIF recording."""
        if self._gif_renderer is None:
            self._gif_renderer = mujoco.Renderer(
                self.env.model, height=480, width=640
            )
        camera = mujoco.MjvCamera()
        camera.type      = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.env.pos[0]
        camera.distance  = 3.5
        camera.azimuth   = -60
        camera.elevation = -20
        self._gif_renderer.update_scene(self.env.data, camera)
        return self._gif_renderer.render()

    def save_gif(self, filename):
        """Saves accumulated frames to a GIF file."""
        if not self.gif_frames:
            return
        from PIL import Image as PILImage
        images = [PILImage.fromarray(f) for f in self.gif_frames]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=30,
            loop=0,
        )
        print(f"\n[GIF] Saved to: {os.path.abspath(filename)}\n")
        self.gif_frames = []

    def render_fpv_hud(self):
        """Renders the drone FPV camera with YOLO overlay (OpenCV window)."""
        if self._fpv_renderer is None:
            self._fpv_renderer = mujoco.Renderer(
                self.env.model, height=self.img_h, width=self.img_w
            )
        self._fpv_renderer.update_scene(self.env.data, self.cam_name)
        img = self._fpv_renderer.render()

        import cv2
        hud_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cx, cy  = self.img_w // 2, self.img_h // 2
        cv2.drawMarker(hud_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)

        yolo_box = self._compute_fake_yolo()
        if yolo_box[0] >= 0.0:
            x_min = int(yolo_box[0] * self.img_w)
            y_min = int(yolo_box[1] * self.img_h)
            x_max = int(yolo_box[2] * self.img_w)
            y_max = int(yolo_box[3] * self.img_h)
            cv2.rectangle(hud_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            tx    = (x_min + x_max) // 2
            ty    = (y_min + y_max) // 2
            cv2.line(hud_img, (cx, cy), (tx, ty), (0, 0, 255), 1)
            cv2.circle(hud_img, (tx, ty), 4, (0, 0, 255), -1)
            err_x = (tx / self.img_w) - 0.5
            err_y = (ty / self.img_h) - 0.5
            cv2.putText(hud_img, "GATE LOCKED", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(hud_img, f"Err X: {err_x:+.2f} | Err Y: {err_y:+.2f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(hud_img, "TARGET LOST (SEARCHING...)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow("Drone FPV Target Tracker", hud_img)
        cv2.waitKey(1)

    def close(self):
        import cv2
        cv2.destroyAllWindows()
        super().close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Factory helpers
# ══════════════════════════════════════════════════════════════════════════════
def _build_base_env(gui: bool = False, rank: int = 0) -> Monitor:
    """Instantiate one KRTIAviary + wrapper, ready for vectorisation."""
    use_gui = gui and not SHOW_FPV_GUI
    base_env = KRTIAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        gui=use_gui,
        vision_attributes=True,
        render_mode="human" if use_gui else None,
        initial_xyzs=np.array([[0.92, 24.47, 0.25]]),
        initial_rpys=np.array([[0.0, 0.0, -np.pi / 2]]),
    )
    wrapped = SingleGateTrainingWrapper(base_env, rank=rank)
    return Monitor(wrapped)


def make_headless_env(rank: int = 0):
    """Thunk for headless parallel environment (SubprocVecEnv factory)."""
    def _init():
        return _build_base_env(gui=False, rank=rank)
    return _init


def make_env(gui: bool = False, rank: int = 0):
    """Thunk for environment with optional GUI (for debug / demo)."""
    def _init():
        return _build_base_env(gui=gui, rank=rank)
    return _init


# ══════════════════════════════════════════════════════════════════════════════
# 7. Curriculum configuration  (mirrored from train_single_rl_jax.py)
# ══════════════════════════════════════════════════════════════════════════════
CURRICULUM_STAGES = [
    {"level": 1, "steps": 3_000_000, "lr": 3.0e-4},  # Fixed config, low speed
    {"level": 2, "steps": 3_000_000, "lr": 2.5e-4},  # Minor variations
    {"level": 3, "steps": 4_000_000, "lr": 2.0e-4},  # Moderate variations
    {"level": 4, "steps": 4_000_000, "lr": 1.5e-4},  # Camera noise, aggressive offset
    {"level": 5, "steps": 6_000_000, "lr": 1.0e-4},  # Full domain randomisation
]


# ══════════════════════════════════════════════════════════════════════════════
# 8. Custom TensorBoard callback (bridges SBX -> torch SummaryWriter)
# ══════════════════════════════════════════════════════════════════════════════
class TBMetricsCallback(BaseCallback):
    """
    Forwards episode metrics to a torch SummaryWriter so that
    multi-stage global steps are tracked correctly across curriculum stages.
    """

    def __init__(self, writer, global_offset: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.writer        = writer
        self.global_offset = global_offset

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Log episode-level metrics and reward components after each rollout."""
        if self.writer is None:
            return
        global_step = self.global_offset + self.num_timesteps
        for info in self.locals.get("infos", []):
            # SB3 Monitor wrapper standard episode summary
            ep_info = info.get("episode")
            if ep_info:
                self.writer.add_scalar(
                    "curriculum/episode_reward", ep_info["r"], global_step
                )
                self.writer.add_scalar(
                    "curriculum/episode_length", ep_info["l"], global_step
                )

            # Custom per-component metrics injected by SingleGateTrainingWrapper
            ep_metrics = info.get("ep_metrics")
            if ep_metrics:
                for tag, value in ep_metrics.items():
                    self.writer.add_scalar(tag, float(value), global_step)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Training entry-point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Stable-Baselines Jax (SBX) -- KRTI Single-Gate Curriculum")
    all_devices = jax.devices()
    gpu_devices = [d for d in all_devices if "cuda" in str(d).lower() or "gpu" in str(d).lower()]
    print(f"  JAX backend:  {jax.default_backend()}")
    print(f"  JAX devices:  {all_devices}")
    if gpu_devices:
        print(f"  ✓ GPU detected: {gpu_devices[0]}  (CUDA acceleration ENABLED)")
    else:
        print("  ⚠ No GPU detected — running on CPU. Install 'jax[cuda12]' for GPU support.")
    print(f"  Parallel environments: {NUM_ENVS}")
    print("=" * 60)

    output_directory = "./results/krti_single_rl_sbx/"
    os.makedirs(output_directory, exist_ok=True)

    checkpoint_dir = os.path.join(output_directory, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── TensorBoard setup ────────────────────────────────────────────────────
    tb_dir = os.path.join(output_directory, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    run_idx = 1
    while os.path.exists(os.path.join(tb_dir, f"run_{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(tb_dir, f"run_{run_idx}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(run_dir)
        print(f"[TB] Logging to: {run_dir}")
    except ImportError:
        print("[TB] torch not found -- TensorBoard metrics use SBX built-in logger.")
        tb_writer = None

    # ── Curriculum loop ──────────────────────────────────────────────────────
    global_step_counter = 0
    model_path_prev     = None  # Path to the most recently saved stage

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        level = stage["level"]
        steps = stage["steps"]
        lr    = stage["lr"]

        print(f"\n{'#' * 60}")
        print(f"  CURRICULUM STAGE {level}  |  {steps:,} steps  |  LR = {lr}")
        print(f"{'#' * 60}\n")

        # Build the vectorised environment cluster
        if DEBUG_GUI:
            env_cluster = DummyVecEnv([make_env(gui=True, rank=0)])
        else:
            env_cluster = SubprocVecEnv(
                [make_headless_env(rank=i) for i in range(NUM_ENVS)]
            )

        stage_tb_log      = os.path.join(output_directory, "tensorboard")
        stage_ckpt_prefix = f"sbx_stage{level}_brain"

        # ── Load or create the SBX PPO model ─────────────────────────────────
        if model_path_prev and os.path.exists(model_path_prev + ".zip"):
            print(f"[TRANSFER] Loading stage-{level - 1} model from "
                  f"{model_path_prev}.zip ...\n")
            model = PPO.load(
                model_path_prev,
                env=env_cluster,
                tensorboard_log=stage_tb_log,
                custom_objects={
                    "learning_rate": lr,
                    "n_steps":       4096,
                    "batch_size":    512,
                },
            )
            model.learning_rate = lr
        else:
            print("[START FRESH] No prior model found -- initialising from scratch.\n")
            model = PPO(
                "MlpPolicy",
                env_cluster,
                learning_rate=lr,
                n_steps=4096,
                batch_size=512,
                n_epochs=10,
                gamma=0.99,
                verbose=1,
                tensorboard_log=stage_tb_log,
                policy_kwargs=dict(
                    # Network architecture matching v2 policies for weight compatibility
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                ),
            )

        # ── Callbacks ────────────────────────────────────────────────────────
        checkpoint_cb = CheckpointCallback(
            save_freq=max(15_000 // NUM_ENVS, 1),
            save_path=checkpoint_dir,
            name_prefix=stage_ckpt_prefix,
        )
        tb_cb = TBMetricsCallback(
            writer=tb_writer,
            global_offset=global_step_counter,
        )

        # ── Train ─────────────────────────────────────────────────────────────
        model.learn(
            total_timesteps=steps,
            callback=[checkpoint_cb, tb_cb],
            tb_log_name=f"PPO_stage{level}",
            reset_num_timesteps=(stage_idx == 0),  # Only reset counter at stage 1
            progress_bar=True,
        )

        # ── Save stage final weights ──────────────────────────────────────────
        stage_final_path = os.path.join(
            output_directory, f"stage_{level}_final_sbx_brain"
        )
        model.save(stage_final_path)
        model_path_prev = stage_final_path

        print(f"\n{'=' * 60}")
        print(f"  STAGE {level} COMPLETE  |  Saved: {stage_final_path}.zip")
        print(f"{'=' * 60}")

        global_step_counter += steps
        env_cluster.close()

        # ── Interactive continuation prompt (same as train_single_rl_jax.py) ─
        if stage_idx < len(CURRICULUM_STAGES) - 1:
            while True:
                choice = input(
                    f"\nAdvance to curriculum stage {level + 1}? "
                    f"[y]es / [n]o / [q]uit: "
                ).strip().lower()
                if choice in ("y", "yes"):
                    print(f"\nInitialising stage {level + 1} ...\n")
                    break
                elif choice in ("n", "no", "q", "quit"):
                    print(
                        f"\nExiting. Progress up to stage {level} is safely saved at:\n"
                        f"  {stage_final_path}.zip\n"
                    )
                    if tb_writer:
                        tb_writer.close()
                    sys.exit(0)
                else:
                    print("Invalid input. Please type 'y' to continue or 'n' to stop.")

    # ── Final combined save ───────────────────────────────────────────────────
    final_path = os.path.join(output_directory, "final_krti_sbx_brain")
    model.save(final_path)
    print(f"\n[DONE] Full curriculum complete. Final model saved to:\n  {final_path}.zip\n")

    if tb_writer:
        tb_writer.close()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if DEBUG_GUI:
        import time
        env_cluster = DummyVecEnv([make_env(gui=True, rank=0)])
        env         = env_cluster.envs[0]
        env.reset()
        if not SHOW_FPV_GUI:
            env.render()
        time.sleep(5)
        env_cluster.close()
    else:
        main()
