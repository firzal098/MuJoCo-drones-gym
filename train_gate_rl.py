import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom arena properties from your source script
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType
from multi_drone_mujoco.examples.krti_arena import KRTIAviary, get_gate_corners, project_point

class TwoGateTrainingWrapper(gym.Wrapper):
    """
    Custom wrapper targeting the first two gates of the KRTI 2026 arena.
    Uses purely geometric equations to simulate noisy YOLO observations headlessly.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 1. Ground Truth World Coordinates of the target gates
        self.gate_targets = [
            {"name": "gate_single_a", "pos": np.array([0.17, 9.26, 1.0]), "type": "single"},
            {"name": "gate_double_a", "pos": np.array([-8.33, 21.1, 1.0]), "type": "double"}
        ]
        self.active_gate_idx = 0
        self.max_episode_steps = 800
        self.current_step = 0
        
        # 2. Add fixed global estimation noise to the gate coordinates (Simulates rough map data)
        # This noise is constant per episode so the agent must rely on vision data to correct course
        self.noisy_gate_positions = []
        for gate in self.gate_targets:
            map_noise = np.random.normal(0, 0.4, size=3) # Up to 40cm initial positioning error
            map_noise[2] = 0.0 # Keep altitude estimates flat
            self.noisy_gate_positions.append(gate["pos"] + map_noise)

        # Actions: [V_forward, V_lateral, V_vertical, Yaw_Rate] bounded between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observations: [YOLO_Box(4), Rel_Noisy_Gate1(3), Rel_Noisy_Gate2(3), EKF_Vel(3), Attitude(3)] = 16 Dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # Cache camera parameters from MuJoCo configuration
        self.cam_name = "drone0_cam"
        self.cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        self.img_w, self.img_h = 320, 240
        self.fovy = self.env.model.cam_fovy[self.cam_id]

    def _compute_fake_yolo(self):
        """Calculates 2D screen bounding boxes mathematically without pixel rendering."""
        gate = self.gate_targets[self.active_gate_idx]
        body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, gate["name"])
        
        if body_id < 0:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        # Extract real-time camera position matrices
        cam_pos = self.env.data.cam_xpos[self.cam_id].copy()
        cam_mat = self.env.data.cam_xmat[self.cam_id].copy()

        T_gate = self.env.data.xpos[body_id].copy()
        R_gate = self.env.data.xmat[body_id].copy().reshape(3, 3)

        local_corners = get_gate_corners(gate["type"])
        px_list = []
        
        for pt_local in local_corners:
            pt_world = R_gate @ pt_local + T_gate
            px = project_point(pt_world, cam_pos, cam_mat, self.fovy, self.img_w, self.img_h)
            if px is not None:
                px_list.append(px)

        if len(px_list) < 2:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32) # Out of sight

        xs = [p[0] for p in px_list]
        ys = [p[1] for p in px_list]

        # Normalize pixel bounding boxes between 0.0 and 1.0
        x_min = np.clip(min(xs) / self.img_w, 0.0, 1.0)
        x_max = np.clip(max(xs) / self.img_w, 0.0, 1.0)
        y_min = np.clip(min(ys) / self.img_h, 0.0, 1.0)
        y_max = np.clip(max(ys) / self.img_h, 0.0, 1.0)

        # Inject artificial vision tracking jitter/noise
        yolo_noise = np.random.normal(0, 0.015, size=4)
        noisy_box = np.array([x_min, y_min, x_max, y_max]) + yolo_noise
        
        # 3. Simulate casual detection drops (5% chance of lost frames)
        if np.random.rand() < 0.05:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        return np.clip(noisy_box, 0.0, 1.0).astype(np.float32)

    def _get_obs(self):
        drone_pos = self.env.pos[0]
        
        # Calculate standard relative vectors from noisy map memory estimates
        rel_gate1 = self.noisy_gate_positions[0] - drone_pos
        rel_gate2 = self.noisy_gate_positions[1] - drone_pos
        
        # Fetch EKF flight dynamics
        vx, vy, vz = self.env.vel[0]
        roll, pitch, yaw = self.env.rpy[0]
        
        # Calculate simulated real-time vision bounding box
        yolo_box = self._compute_fake_yolo()

        return np.concatenate([
            yolo_box,
            rel_gate1,
            rel_gate2,
            [vx, vy, vz],
            [roll, pitch, yaw]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.active_gate_idx = 0
        self.current_step = 0
        
        # Regenerate noisy estimates each episode initialization block
        self.noisy_gate_positions = []
        for gate in self.gate_targets:
            map_noise = np.random.normal(0, 0.35, size=3)
            map_noise[2] = 0.0 
            self.noisy_gate_positions.append(gate["pos"] + map_noise)
            
        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1
        
        # Translate action scaling bounds to local body flight targets
        max_speed = 5.0      # meters per second
        max_yaw_rate = 1.2   # radians per second

        # Map actions to continuous high-level control states in body frame
        vx_body = action[0] * max_speed
        vy_body = action[1] * max_speed
        vz_body = action[2] * max_speed
        yaw_rate = action[3] * max_yaw_rate

        # Rotate body velocity to world coordinates for KRTIAviary ActionType.VEL
        yaw = self.env.rpy[0, 2]
        vx_world = vx_body * np.cos(yaw) - vy_body * np.sin(yaw)
        vy_world = vx_body * np.sin(yaw) + vy_body * np.cos(yaw)
        vz_world = vz_body  # vertical is same

        scaled_action = np.array([vx_world, vy_world, vz_world, yaw_rate], dtype=np.float32)

        target_gate = self.gate_targets[self.active_gate_idx]["pos"]
        dist_before = np.linalg.norm(target_gate - self.env.pos[0])

        # Execute step commands through the env (which expects world velocity)
        _, _, _, _, info = self.env.step(scaled_action)

        dist_after = np.linalg.norm(target_gate - self.env.pos[0])

        # 1. Trajectory Progression Reward Matrix
        reward = (dist_before - dist_after) * 20.0

        terminated = False
        truncated = False

        # 2. Threshold Detection Check (Successful Gate Passage)
        if dist_after < 0.40:
            reward += 150.0  # Gate clearance bonus
            if self.active_gate_idx == 0:
                self.active_gate_idx = 1 # Shift focus target matrix to Gate 2
                print(" --> [SUCCESS] Cleared Gate 1! Re-routing to Gate 2.")
            else:
                reward += 300.0  # Completed mission sequence criteria
                print(" --> [MISSION COMPLETE] Cleared both target gates successfully!")
                terminated = True

        # 3. Structural Survival Safety Filters
        rpy = self.env.rpy[0]
        has_crashed = (
            self.env.pos[0, 2] < 0.08 or 
            self.env.pos[0, 2] > 4.5 or 
            abs(rpy[0]) > np.pi/2.5 or 
            abs(rpy[1]) > np.pi/2.5
        )

        if has_crashed:
            reward -= 200.0  # Heavy crash penalty
            terminated = True

        # 4. Action Efficiency / Stabilization Constraints
        reward -= 0.05 * np.sum(np.square(action))

        if self.current_step >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

def make_headless_env():
    def _init():
        # Clean background layers completely to scale up step frequency loops
        return TwoGateTrainingWrapper(KRTIAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.MJC,
            sim_freq=240,
            ctrl_freq=48,
            act_type=ActionType.VEL,  # Train explicitly on Velocity Space vectors
            gui=False,                # Headless optimization active
            vision_attributes=True,   # Keep camera coordinates compile-time active
            render_mode=None
        ))
    return _init

if __name__ == "__main__":
    output_directory = "./results/krti_gate_rl/"
    os.makedirs(output_directory, exist_ok=True)

    # Launch 4 synchronous lightweight math environments in parallel
    num_envs = 4
    env_cluster = SubprocVecEnv([make_headless_env() for _ in range(num_envs)])

    print("=" * 60)
    print("Launching Headless Multi-Gate Training System")
    print(f"  Parallel instances active: {num_envs}")
    print("=" * 60)

    model = PPO(
        "MlpPolicy",
        env_cluster,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=os.path.join(output_directory, "tensorboard"),
        device="cpu"
    )

    checkpoint_tracker = CheckpointCallback(
        save_freq=15_000,
        save_path=output_directory,
        name_prefix="krti_vtol_brain"
    )

    # Execute 800,000 steps of headless interaction optimization
    model.learn(total_timesteps=800_000, callback=checkpoint_tracker)
    model.save(os.path.join(output_directory, "final_krti_vtol_brain"))
    print("Optimization tracking sequence complete.")
