import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor  # <-- Add this import at the top

# Import custom aviary/arena components
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType
from multi_drone_mujoco.examples.krti_arena import KRTIAviary, get_gate_corners, project_point
from multi_drone_mujoco.control.guided_mode import GuidedVehicle

DEBUG_GUI = False
RECORD_GIF = False  # Set to True to record episodes and save them as a GIF
SHOW_FPV_GUI = False  # Set to True to spawn a custom FPV target tracker window instead of the standard MuJoCo GUI
NUM_ENVS = 1 if (DEBUG_GUI or SHOW_FPV_GUI) else 6

class SingleGateTrainingWrapper(gym.Wrapper):
    """
    Custom wrapper targeting strictly gate_single_a of the KRTI 2026 arena.
    Includes domain randomization (gate and drone spawns) and target heading tracking.
    """
    def __init__(self, env, rank=0):
        super().__init__(env)
        self.rank = rank
        
        # 1. Ground Truth Nominal Position of gate_single_a
        self.nominal_gate_pos = np.array([0.17, 12.26, 1.0])
        self.gate_targets = [
            {"name": "gate_single_a", "pos": self.nominal_gate_pos.copy(), "type": "single"}
        ]
        self.max_episode_steps = 600
        self.current_step = 0
        
        # 2. Map estimation noise initialized placeholder (overwritten at reset)
        self.noisy_gate_position = self.nominal_gate_pos.copy()

        # Actions: [V_forward, V_lateral, V_vertical_down, Yaw_Rate] bounded between -1.0 and 1.0 (Option B)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Instantiate GuidedVehicle for ArduPilot GUIDED mode simulation
        self.vehicle = GuidedVehicle(self.env, drone_index=0)

        # GIF Recording parameters
        self.record_gif = RECORD_GIF
        self.gif_frames = []
        self._gif_renderer = None

        # FPV HUD window parameters
        self.show_fpv_gui = SHOW_FPV_GUI
        self._fpv_renderer = None

        # Observations: [YOLO_Box(4), Rel_Noisy_Gate(3), EKF_Vel(3), Attitude(3), Padding(87)] = 100 Dimensions
        self.observation_space = spaces.Box(low=-1.2, high=1.2, shape=(100,), dtype=np.float32)

        # Cache camera parameters from MuJoCo configuration
        self.cam_name = "drone0_cam"
        self.cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        self.img_w, self.img_h = 320, 240
        self.fovy = self.env.model.cam_fovy[self.cam_id]

    def _compute_fake_yolo(self):
        """Calculates 2D screen bounding boxes mathematically without pixel rendering."""
        gate = self.gate_targets[0]
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
                px_x, px_y = px
                # Only count the corner if it actually falls within the camera frame boundaries
                if 0 <= px_x <= self.img_w and 0 <= px_y <= self.img_h:
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
        
        # 5% chance of visual tracking frame drops
        if np.random.rand() < 0.05:
            return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        return np.clip(noisy_box, 0.0, 1.0).astype(np.float32)

    def _get_obs(self):
        drone_pos = self.env.pos[0]
        
        # Fetch EKF flight dynamics in world coordinates
        vx_world, vy_world, vz_world = self.env.vel[0]
        roll, pitch, yaw = self.env.rpy[0]
        
        # Rotate relative gate vector into the drone's body frame
        rel_gate_world = self.noisy_gate_position - drone_pos
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        rel_gate_body = np.array([
            rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,  # Forward
            -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y), # Right (FRD)
            -rel_gate_world[2]                                      # Down (FRD)
        ])
        
        # Rotate EKF velocities into the drone's body frame
        vel_body = np.array([
            vx_world * cos_y + vy_world * sin_y,                    # Forward
            -(-vx_world * sin_y + vy_world * cos_y),                   # Right (FRD)
            -vz_world                                               # Down (FRD)
        ])
        
        # Calculate simulated vision bounding box in [err_x, err_y, box_w, box_h] format
        yolo_raw = self._compute_fake_yolo()
        if yolo_raw[0] >= 0.0:
            x_min, y_min, x_max, y_max = yolo_raw
            err_x = ((x_min + x_max) / 2.0) - 0.5
            err_y = ((y_min + y_max) / 2.0) - 0.5
            box_w = x_max - x_min
            box_h = y_max - y_min
            yolo_obs = np.array([err_x, err_y, box_w, box_h], dtype=np.float32)
        else:
            yolo_obs = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        yaw_error = np.arctan2(rel_gate_body[1], rel_gate_body[0])
        
        # Scale inputs for PPO [-1, 1] range
        rel_gate_body_scaled = np.clip(rel_gate_body / 20.0, -1.0, 1.0)
        vel_body_scaled = np.clip(vel_body / 4.0, -1.0, 1.0)
        
        # Convert MuJoCo pitch (Nose Down positive) to FRD pitch (Nose Up positive)
        pitch_frd = -pitch
        attitude_scaled = np.clip(np.array([roll, pitch_frd, yaw_error]) / np.pi, -1.0, 1.0)
        
        padding = np.zeros(87, dtype=np.float32)

        return np.concatenate([
            yolo_obs,
            rel_gate_body_scaled,
            vel_body_scaled,
            attitude_scaled,
            padding
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        # 1. Domain Randomization - Gate Position in MuJoCo model
        # Randomize nominal distance (Y coordinate) between 8.0m and 18.0m
        base_gate_y = np.random.uniform(8.0, 18.0)
        self.nominal_gate_pos = np.array([0.17, base_gate_y, 1.0])

        # Randomize gate offset by up to +/- 1.0m in X and Y
        gx = np.random.uniform(-1.0, 1.0)
        gy = np.random.uniform(-1.0, 1.0)
        
        body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")
        if body_id >= 0:
            self.env.model.body_pos[body_id] = np.array([0.17 + gx, base_gate_y + gy, 0.0])
            
        self.gate_targets[0]["pos"] = self.nominal_gate_pos + np.array([gx, gy, 0.0])

        # 2. Domain Randomization - Drone Spawn Position
        # Randomize spawn offset around start zone by up to +/- 1.0m in X and Y
        dx = np.random.uniform(-1.0, 1.0)
        dy = np.random.uniform(-1.0, 1.0)
        self.env.INIT_XYZS[0] = np.array([0.92 + dx, 24.47 + dy, 0.25])

        # 3. Domain Randomization - Drone Spawn Heading
        dyaw = np.random.uniform(-0.3, 0.3)
        self.env.INIT_RPYS[0] = np.array([0.0, 0.0, -np.pi/2 + dyaw])

        # Call the base environment reset which applies INIT_XYZS and INIT_RPYS
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset GIF frames buffer at the start of each episode
        if self.record_gif:
            self.gif_frames = []
            self.gif_frames.append(self.capture_frame())

        # Reset and arm GuidedVehicle
        self.vehicle.disarm()
        self.vehicle.arm()
        
        # Take off and stabilize at 1.0m altitude
        self.vehicle.simple_takeoff(1.0)
        
        # Run takeoff loop dynamically
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
            
        # Settle in hover for 20 steps
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
        self.current_step = 0
        
        # 4. Generate the noisy estimate of the target gate position (simulates rough mapping data)
        # Up to 20cm initial positioning noise (e.g. RTK GPS/Visual SLAM mapping accuracy)
        map_noise = np.random.normal(0, 0.20, size=3)
        map_noise[2] = 0.0
        self.noisy_gate_position = self.gate_targets[0]["pos"] + map_noise

        return self._get_obs(), info

    def capture_frame(self):
        """Captures a 3D tracking frame of the drone for GIF recording."""
        if self._gif_renderer is None:
            self._gif_renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        # Track drone 0 position
        drone_pos = self.env.pos[0]
        camera.lookat[:] = drone_pos
        camera.distance = 3.5
        camera.azimuth = -60
        camera.elevation = -20
        
        self._gif_renderer.update_scene(self.env.data, camera)
        return self._gif_renderer.render()

    def save_gif(self, filename):
        """Saves accumulated frames to a GIF file."""
        if not self.gif_frames:
            return
        from PIL import Image as PILImage
        images = [PILImage.fromarray(f) for f in self.gif_frames]
        # Ensure target folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=30,  # ~33 FPS
            loop=0
        )
        print(f"\n[GIF RECORDING] Saved latest trajectory GIF to: {os.path.abspath(filename)}\n")
        self.gif_frames = []

    def render_fpv_hud(self):
        """Renders the front camera view of the drone with YOLO box and center error overlays."""
        if self._fpv_renderer is None:
            self._fpv_renderer = mujoco.Renderer(self.env.model, height=self.img_h, width=self.img_w)
        
        # Render camera view
        self._fpv_renderer.update_scene(self.env.data, self.cam_name)
        img = self._fpv_renderer.render()
        
        import cv2
        # Convert RGB to BGR for OpenCV
        hud_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Center of the screen
        cx, cy = self.img_w // 2, self.img_h // 2
        
        # Draw camera center crosshair
        cv2.drawMarker(hud_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=1)
        
        # Get current YOLO bounding box
        yolo_box = self._compute_fake_yolo()
        
        if yolo_box[0] >= 0.0:
            # Map normalized coordinates [0, 1] to pixel space
            x_min = int(yolo_box[0] * self.img_w)
            y_min = int(yolo_box[1] * self.img_h)
            x_max = int(yolo_box[2] * self.img_w)
            y_max = int(yolo_box[3] * self.img_h)
            
            # Draw YOLO bounding box (green)
            cv2.rectangle(hud_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Calculate target center
            tx = (x_min + x_max) // 2
            ty = (y_min + y_max) // 2
            
            # Draw line from camera center to gate center (red)
            cv2.line(hud_img, (cx, cy), (tx, ty), (0, 0, 255), 1)
            cv2.circle(hud_img, (tx, ty), 4, (0, 0, 255), -1)
            
            # Calculate center errors
            err_x = (tx / self.img_w) - 0.5
            err_y = (ty / self.img_h) - 0.5
            
            # Overlay text values
            cv2.putText(hud_img, "GATE LOCKED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(hud_img, f"Err X: {err_x:+.2f} | Err Y: {err_y:+.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            # Bounding box is not visible
            cv2.putText(hud_img, "TARGET LOST (SEARCHING...)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
        cv2.imshow("Drone FPV Target Tracker", hud_img)
        cv2.waitKey(1)

    def close(self):
        """Cleans up OpenCV GUI windows upon environment shutdown."""
        import cv2
        cv2.destroyAllWindows()
        super().close()

    def _check_gate_collision(self):
        """Checks if the drone has collided with the gate geoms."""
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            body1_id = self.env.model.geom_bodyid[contact.geom1]
            body2_id = self.env.model.geom_bodyid[contact.geom2]
            name1 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            name2 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            is_drone = (name1 is not None and "drone0" in name1) or (name2 is not None and "drone0" in name2)
            is_gate = (
                (name1 is not None and "gate_" in name1) or 
                (name2 is not None and "gate_" in name2)
            )
            if is_drone and is_gate:
                return True
        return False

    def step(self, action):
        self.current_step += 1
        
        max_xy_speed = 3.0     # meters per second
        max_z_speed = 1.5      # meters per second
        max_yaw_rate = 1.8   # radians per second

        # Map actions: The Neural Network now natively outputs FRD!
        # [vx_body (Forward), vy_body (Right), vz_body (Down), yaw_rate (CW)]
        
        # Map actions from FRD (Neural Network) to FLU (GuidedVehicle/MuJoCo)
        # [vx_body (Forward), vy_body (Right->Left), vz_body (Down->Up), yaw_rate (CW->CCW)]
        vx_body = action[0] * max_xy_speed
        vy_body = -action[1] * max_xy_speed
        vz_body = -action[2] * max_z_speed
        yaw_rate = -action[3] * max_yaw_rate

        # Update GuidedVehicle with target velocities
        self.vehicle.set_velocity(vx_body, vy_body, vz_body, yaw_rate)
        rpm = self.vehicle.update(self.env.CTRL_TIMESTEP)

        target_gate = self.gate_targets[0]["pos"]
        dist_before = np.linalg.norm(target_gate - self.env.pos[0])

        # Step the environment with computed RPMs
        _, _, _, _, info = self.env.step(rpm)

        # Sync the passive viewer if rendering is active
        if self.env.render_mode == "human":
            self.env.render()
        if self.show_fpv_gui:
            self.render_fpv_hud()

        dist_after = np.linalg.norm(target_gate - self.env.pos[0])

        # 1. Trajectory Progression Reward Matrix
        reward = (dist_before - dist_after) * 20.0

        # Continuous Heading Alignment Penalty (Yaw Error)
        drone_pos = self.env.pos[0]
        _, _, yaw = self.env.rpy[0]
        rel_gate_world = self.noisy_gate_position - drone_pos
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        
        # Calculate in FRD to match observations exactly
        rel_gate_body_x = rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y
        rel_gate_body_y = -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y) # Right (FRD)
        
        yaw_error = np.arctan2(rel_gate_body_y, rel_gate_body_x)
        reward -= 0.15 * abs(yaw_error)

        # Continuous Vertical Alignment Penalty (Height Error)
        height_error = -rel_gate_world[2] # Down (FRD)
        reward -= 0.15 * abs(height_error)

        # Alignment error penalty (penalize displacement of the gate from camera center)
        yolo_box = self._compute_fake_yolo()
        if yolo_box[0] >= 0.0:  # Target gate is visible in front camera view
            x_center = (yolo_box[0] + yolo_box[2]) / 2.0
            y_center = (yolo_box[1] + yolo_box[3]) / 2.0
            err_x = x_center - 0.5
            err_y = y_center - 0.5
            alignment_penalty = 1.0 * np.sqrt(err_x**2 + err_y**2)
            reward -= alignment_penalty
        else:
            # Penalize losing visual contact with target gate
            reward -= 0.20

        terminated = False
        truncated = False

        # 2. Threshold Detection Check (Successful Gate Passage)
        # Use 0.25 to ensure we catch the pass, but reward it heavily for being near 0.0!
        if dist_after < 0.25:
            center_accuracy_bonus = (0.25 - dist_after) * 800.0  # Up to +200 bonus for perfect center
            reward += 200.0 + center_accuracy_bonus
            if self.show_fpv_gui or self.record_gif or self.rank == 0:
                print(f" --> [MISSION COMPLETE] Cleared Gate! (Accuracy Bonus: +{center_accuracy_bonus:.1f})")
            terminated = True

        # Check collision with gate
        if self._check_gate_collision():
            reward -= 450.0  # Heavy penalty for gate collision
            terminated = True
            if self.show_fpv_gui or self.record_gif or self.rank == 0:
                print(" --> [CRASH] Collided with Gate Single A!")

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
            if self.show_fpv_gui or self.record_gif or self.rank == 0:
                print(" --> [CRASH] Drone hit the ground or flipped!")

        # 4. Action Efficiency / Stabilization Constraints
        reward -= 0.05 * np.sum(np.square(action))

        # 5. Time/Step Penalty to encourage forward progress and avoid cowardly hovering
        reward -= 0.10

        if self.current_step >= self.max_episode_steps:
            truncated = True

        # Capture frame at the end of step
        if self.record_gif:
            self.gif_frames.append(self.capture_frame())

        # Save GIF at the end of the episode (overwriting latest file for convenience)
        if (terminated or truncated) and self.record_gif:
            self.save_gif("./results/trajectory.gif")

        return self._get_obs(), reward, terminated, truncated, info

def make_headless_env(rank=0):
    def _init():

        HEADLESS = True

        # 1. Initialize core environment
        base_env = KRTIAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.MJC,
            sim_freq=240,
            ctrl_freq=48,
            act_type=ActionType.RPM,  
            gui=not HEADLESS,
            vision_attributes=True,
            render_mode=None if HEADLESS else "human",
            initial_xyzs=np.array([[0.92, 24.47, 0.25]]),
            initial_rpys=np.array([[0.0, 0.0, -np.pi/2]]),  
        )
        
        # 2. Wrap with your custom gate/randomization wrapper
        wrapped_env = SingleGateTrainingWrapper(base_env, rank=rank)
        
        # 3. CRITICAL: Wrap with Monitor so SB3 can log rewards to TensorBoard!
        return Monitor(wrapped_env)
    return _init

def make_env(gui=False, rank=0):
    def _init():
        # If FPV GUI is active, suppress standard passive viewer gui
        use_standard_gui = gui and not SHOW_FPV_GUI
        env = KRTIAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.MJC,
            sim_freq=240,
            ctrl_freq=48,
            act_type=ActionType.RPM,
            gui=use_standard_gui,
            vision_attributes=True,
            render_mode="human" if use_standard_gui else None,
            initial_xyzs=np.array([[0.92, 24.47, 0.25]]),
            initial_rpys=np.array([[0.0, 0.0, -np.pi/2]])
        )

        return Monitor(SingleGateTrainingWrapper(env, rank=rank))

    return _init


if __name__ == "__main__":
    output_directory = "./results/krti_single_rl_v2/"
    os.makedirs(output_directory, exist_ok=True)

    # Launch 4 synchronous lightweight math environments in parallel
    if DEBUG_GUI:
        from stable_baselines3.common.vec_env import DummyVecEnv

        env_cluster = DummyVecEnv([make_env(gui=True, rank=0)])
    else:
        env_cluster = SubprocVecEnv(
            [make_env(gui=False, rank=i) for i in range(NUM_ENVS)]
        )

    print("=" * 60)
    print("Launching Headless Single-Gate RL Training System")
    print(f"  Parallel instances active: {NUM_ENVS}")
    print("=" * 60)

    model_path = os.path.join(output_directory, "final_krti_single_brain")
    if os.path.exists(model_path + ".zip"):
        print(f"\n[TRANSFER LEARNING] Loading pre-trained model from {model_path}.zip to continue training...\n")
        model = PPO.load(
            model_path, 
            env=env_cluster, 
            tensorboard_log=os.path.join(output_directory, "tensorboard"),
            custom_objects={
                "n_steps": 4096,
                "batch_size": 512,
                "learning_rate": 3e-4
            }
        )
    else:
        print("\n[START FRESH] No pre-trained model found. Initializing a new PPO model from scratch...\n")
        model = PPO(
            "MlpPolicy",
            env_cluster,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log=os.path.join(output_directory, "tensorboard"),
            device="cpu"
        )

    if DEBUG_GUI:
        env = env_cluster.envs[0]      # DummyVecEnv only
        env.reset()
        if not SHOW_FPV_GUI:
            env.render()

        import time
        time.sleep(5)

    checkpoint_tracker = CheckpointCallback(
        save_freq=15_000,
        save_path=output_directory,
        name_prefix="krti_single_brain"
    )                           

    # Execute 800,000 steps of headless interaction optimization
    model.learn(total_timesteps=3_000_000, callback=checkpoint_tracker)
    model.save(os.path.join(output_directory, "final_krti_single_brain"))
    print("Optimization tracking sequence complete.")
