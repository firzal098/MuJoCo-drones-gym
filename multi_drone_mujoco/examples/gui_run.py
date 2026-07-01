"""Example: Run the MuJoCo simulator as long as the GUI window is open.

This script demonstrates how to keep the simulation loop active for as long
as the user keeps the passive graphical user interface (GUI) window open.
If the drone crashes (terminated), the environment resets to allow continuous running.
"""

import os
# Force OpenGL rendering to use NVIDIA GPU under WSL
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
# Disable V-Sync to prevent frame rate capping and stutter under WSLg
os.environ["vblank_mode"] = "0"
os.environ["__GL_SYNC_TO_VBLANK"] = "0"

import time
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType

def main():
    print("=" * 60)
    print("Indefinite GUI Run Example")
    print("  Controls a single Crazyflie to hover at z = 1.0m.")
    print("  Simulation runs until you close the GUI window.")
    print("=" * 60)

    # 1. Initialize the environment with GUI / human rendering enabled
    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        gui=True,
        render_mode="human"
    )

    # 2. Set up a PID controller and reset the environment
    ctrl = PIDControl(env)
    obs, info = env.reset()
    target_pos = np.array([0.0, 0.0, 1.0])

    # 3. Call render() once to spawn/launch the passive viewer
    env.render()

    print("\nSimulation started. Close the GUI window to exit the script.\n")

    step = 0
    last_render_time = 0.0
    # 4. Loop as long as the GUI window is running
    while env._viewer is not None and env._viewer.is_running():
        start_time = time.time()

        # Compute PID control inputs (motor RPMs)
        rpm, pos_e, yaw_e = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=env.pos[0],
            cur_quat=env.quat[0],
            cur_vel=env.vel[0],
            cur_ang_vel=env.ang_v[0],
            target_pos=target_pos,
        )

        # Step the environment with RPM action
        obs, reward, terminated, truncated, info = env.step(rpm.flatten())

        # Render at most 30 FPS to reduce rendering overhead and lag under WSLg
        current_time = time.time()
        if current_time - last_render_time >= 1.0 / 30.0:
            env.render()
            last_render_time = current_time

        # Print current position every 1 second (48 control steps)
        if step % 48 == 0:
            print(f"t={step * env.CTRL_TIMESTEP:5.1f}s | pos=[{env.pos[0,0]:+.3f}, {env.pos[0,1]:+.3f}, {env.pos[0,2]:+.3f}] | error={np.linalg.norm(pos_e):.4f}")

        step += 1

        # Check for crash or out-of-bounds (height < 0.05m or > 3.0m, or roll/pitch flipped)
        rpy = env.rpy[0]
        has_crashed = env.pos[0, 2] < 0.05 or env.pos[0, 2] > 3.0 or abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2

        if has_crashed:
            print("  [CRASH] Drone crashed or went out of bounds! Resetting environment...")
            obs, info = env.reset()

        # Pace the simulation loop to match real-time
        elapsed = time.time() - start_time
        if elapsed < env.CTRL_TIMESTEP:
            time.sleep(env.CTRL_TIMESTEP - elapsed)

    # 5. Clean up and close the environment
    env.close()
    print("\nViewer closed. Exit successful.")

if __name__ == "__main__":
    main()
