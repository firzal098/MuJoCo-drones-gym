"""Example script demonstrating ArduPilot-like GUIDED mode control in MuJoCo."""

import os
import sys
import time
import numpy as np

# Force OpenGL rendering under WSL
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
os.environ["vblank_mode"] = "0"
os.environ["__GL_SYNC_TO_VBLANK"] = "0"

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.guided_mode import GuidedVehicle
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType

def main():
    USE_GUI = True
    if "--headless" in sys.argv:
        USE_GUI = False

    print("=" * 60)
    print("ArduPilot GUIDED Mode Simulation Example")
    print(f"  Mode: {'GUI Visualizer' if USE_GUI else 'Headless'}")
    print("=" * 60)

    # 1. Initialize environment (using CF2X model)
    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        initial_xyzs=np.array([[0.0, 0.0, 0.05]]),
        gui=USE_GUI,
        render_mode="human" if USE_GUI else None
    )

    # 2. Instantiate our high-level guided wrapper
    vehicle = GuidedVehicle(env, drone_index=0)

    # 3. Arm and Takeoff
    print("\n[USER SCRIPT] Arming vehicle...")
    vehicle.arm()
    
    takeoff_altitude = 1.0  # meters
    print(f"[USER SCRIPT] Commanding takeoff to {takeoff_altitude}m...")
    vehicle.simple_takeoff(takeoff_altitude)

    # Waypoints to traverse after takeoff completes
    waypoints = [
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    wp_idx = 0
    state = "TAKEOFF"

    step = 0
    last_render_time = 0.0
    
    running = True
    while running:
        if USE_GUI:
            if env._viewer is None or not env._viewer.is_running():
                break
        
        start_time = time.time()

        # Update controller and get RPMs
        rpm = vehicle.update(control_timestep=env.CTRL_TIMESTEP)

        # Step simulation environment
        obs, reward, terminated, truncated, info = env.step(rpm)

        # Handle user mission state machine
        if vehicle.armed:
            cur_pos = env.pos[0]
            
            # Once takeoff mode completes, transition to waypoint navigation
            if state == "TAKEOFF" and vehicle.mode == "GUIDED":
                print(f"\n[USER SCRIPT] Takeoff complete! Heading to Waypoint 1: {waypoints[wp_idx]}")
                vehicle.simple_goto(waypoints[wp_idx])
                state = "NAVIGATING"
                
            elif state == "NAVIGATING":
                # Check distance to current waypoint
                dist = np.linalg.norm(cur_pos - waypoints[wp_idx])
                if dist < 0.15:
                    print(f"[USER SCRIPT] Reached Waypoint {wp_idx + 1}!")
                    wp_idx += 1
                    if wp_idx < len(waypoints):
                        print(f"[USER SCRIPT] Heading to Waypoint {wp_idx + 1}: {waypoints[wp_idx]}")
                        vehicle.simple_goto(waypoints[wp_idx])
                    else:
                        print("\n[USER SCRIPT] Completed all waypoints! Commanding landing...")
                        vehicle.land()
                        state = "LANDING"
        else:
            # If disarmed after landing, exit loop
            if state == "LANDING":
                print("[USER SCRIPT] Landed and disarmed successfully. Exiting simulation.")
                running = False
                break
            elif step > 50:
                print("[USER SCRIPT] Vehicle disarmed unexpectedly.")
                running = False
                break

        # Render visualizer
        if USE_GUI:
            current_time = time.time()
            if current_time - last_render_time >= 1.0 / 30.0:
                env.render()
                last_render_time = current_time

        # Print telemetry
        if step % 48 == 0:
            print(f"t={step * env.CTRL_TIMESTEP:5.1f}s | mode={vehicle.mode:10s} | pos=[{env.pos[0,0]:+.2f}, {env.pos[0,1]:+.2f}, {env.pos[0,2]:+.2f}] | target_pos=[{vehicle.target_pos[0]:+.2f}, {vehicle.target_pos[1]:+.2f}, {vehicle.target_pos[2]:+.2f}]")

        step += 1

        # Check crash/bounds limit
        rpy = env.rpy[0]
        has_crashed = env.pos[0, 2] < 0.02 or env.pos[0, 2] > 3.0 or abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2
        if has_crashed and state != "LANDING":
            print(f"[USER SCRIPT] CRASH/OUT-OF-BOUNDS detected at pos={env.pos[0]} rpy={env.rpy[0]}! Exiting...")
            break

        # Pace loop to real-time
        elapsed = time.time() - start_time
        if elapsed < env.CTRL_TIMESTEP:
            time.sleep(env.CTRL_TIMESTEP - elapsed)

    env.close()
    print("Simulation finished.")

if __name__ == "__main__":
    main()
