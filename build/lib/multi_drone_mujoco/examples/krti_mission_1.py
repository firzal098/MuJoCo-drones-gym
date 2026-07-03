"""KRTI Mission 1: Passing Single Gate A & C with Body-Frame Velocity commands and front camera rendering."""

import os
import sys
import time
import queue
import threading
import numpy as np
from PIL import Image

# Force OpenGL rendering under WSL
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
os.environ["vblank_mode"] = "0"
os.environ["__GL_SYNC_TO_VBLANK"] = "0"

import mujoco
from multi_drone_mujoco.examples.krti_arena import KRTIAviary, draw_fake_yolo_boxes
from multi_drone_mujoco.control.guided_mode import GuidedVehicle
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType

def main():
    USE_GUI = True
    if "--headless" in sys.argv:
        USE_GUI = False
    gates = {
        "gate_single_a": "single",
        "gate_single_b": "single",
        "gate_single_c": "single",
        "gate_double_a": "double",
        "gate_triple_a": "triple",
    }
    print("=" * 60)
    print("KRTI Mission 1: Passing Single Gate A & C (Body Velocity + Camera)")
    print(f"  Mode: {'GUI Visualizer' if USE_GUI else 'Headless'}")
    print("=" * 60)

    # 1. Initialize environment (reusing custom KRTIAviary)
    env = KRTIAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        gui=USE_GUI,
        vision_attributes=True,
        render_mode="human" if USE_GUI else None
    )

    # 2. Initialize GuidedVehicle wrapper
    vehicle = GuidedVehicle(env, drone_index=0)

    # 3. Mission Waypoint sequence (Start Zone -> Single Gate A -> Single Gate C)
    waypoints = [
        (np.array([0.92, 24.47, 1.0]), "Hover above Start Zone"),
        (np.array([1.12, 9.26, 1.0]), "Single Gate A (Fly-through)"),
        (np.array([-1.98, 14.59, 1.0]), "Single Gate C (Fly-through)"),
    ]
    wp_idx = 0
    target_pos = waypoints[wp_idx][0]
    wp_name = waypoints[wp_idx][1]

    # --- Live front camera display setup ---
    pygame_ok = False
    display_scale = 2
    cam_width, cam_height = 320, 240
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((cam_width * display_scale, cam_height * display_scale))
        pygame.display.set_caption("Live Drone Front Camera (YOLO Gate Detection)")
        pygame_ok = True
        print("Successfully initialized live front camera display window.")
    except Exception as e:
        print(f"Could not initialize Pygame window: {e}. Running in headless/save-only mode.")

    # Find the camera ID for drone0_cam
    cam_name = "drone0_cam"
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        print("Warning: drone0_cam not found in environment!")

    # Video recording container
    frames = []

    # Shared queue for camera thread snapping
    _cam_frame_queue: queue.Queue = queue.Queue(maxsize=2)
    _latest_frame = {"rgb": None, "lock": threading.Lock()}

    def _camera_thread_fn():
        """Background thread: renders front-camera frames at up to 60 FPS without blocking the sim."""
        nonlocal frames
        local_renderer = mujoco.Renderer(env.model, height=cam_height, width=cam_width)
        local_data = mujoco.MjData(env.model)
        target_dt = 1.0 / 60.0

        while True:
            t0 = time.perf_counter()
            try:
                data_snapshot = _cam_frame_queue.get(timeout=0.5)
            except queue.Empty:
                if not threading.current_thread().daemon:
                    break
                continue
            if data_snapshot is None:  # sentinel → exit
                break

            # Copy state into local data
            mujoco.mj_copyData(local_data, env.model, data_snapshot)

            # Render RGB frame
            local_renderer.update_scene(local_data, camera=cam_name)
            rgb = local_renderer.render()

            img = Image.fromarray(rgb)
            draw_fake_yolo_boxes(img, env.model, local_data, cam_id, gates, cam_width, cam_height)
            frames.append(np.array(img))

            with _latest_frame["lock"]:
                _latest_frame["rgb"] = np.array(img)

            # Pace to target FPS
            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    cam_thread = threading.Thread(target=_camera_thread_fn, daemon=True)
    cam_thread.start()
    
    # Arm and Takeoff
    print("\n[MISSION] Arming vehicle...")
    vehicle.arm()
    
    print(f"[MISSION] Commanding takeoff to target: {wp_name} at {target_pos}...")
    vehicle.simple_takeoff(target_pos[2])

    state = "TAKEOFF"
    step = 0
    last_render_time = 0.0
    hover_start_time = None
    
    loop_step_count = 0
    render_count = 0
    last_fps_time = time.time()

    # Call render once to spawn passive viewer
    if USE_GUI:
        env.render()

    running = True
    while running:
        if USE_GUI:
            if env._viewer is None or not env._viewer.is_running():
                break

        start_time = time.time()
        loop_step_count += 1

        # Mission state machine & velocity setpoint computation
        dist = np.linalg.norm(env.pos[0] - target_pos)

        if vehicle.armed:
            if state == "TAKEOFF" and vehicle.mode == "GUIDED":
                # Takeoff complete, transition to navigating
                print(f"  [SUCCESS] Reached takeoff point: {wp_name}!")
                wp_idx += 1
                target_pos = waypoints[wp_idx][0]
                wp_name = waypoints[wp_idx][1]
                print(f"\n[MISSION] Heading to Waypoint {wp_idx + 1}/{len(waypoints)}: {wp_name} at {target_pos}...")
                state = "NAVIGATING"
                
            elif state == "NAVIGATING":
                # Check if current waypoint reached
                if dist < 0.25:
                    print(f"  [SUCCESS] Reached Waypoint {wp_idx + 1}: {wp_name}!")
                    wp_idx += 1
                    if wp_idx < len(waypoints):
                        target_pos = waypoints[wp_idx][0]
                        wp_name = waypoints[wp_idx][1]
                        print(f"\n[MISSION] Heading to Waypoint {wp_idx + 1}/{len(waypoints)}: {wp_name} at {target_pos}...")
                    else:
                        print("\n[MISSION] All target gates passed successfully! Commencing safety brake hover.")
                        state = "HOVER"
                        hover_start_time = time.time()
            
            elif state == "HOVER":
                if hover_start_time is not None and time.time() - hover_start_time > 4.0:
                    print("[MISSION] Hover duration complete. Landing vehicle...")
                    vehicle.land()
                    state = "LANDING"
        else:
            # If disarmed, stop
            if state == "LANDING":
                print("[MISSION] Disarmed after landing. Exiting.")
                running = False
                break

        # Compute and send velocity commands
        if state == "NAVIGATING":
            # 1. Proportional position control to get desired world velocity vector
            max_speed = 3.5
            kp_pos = 1.2
            target_speed = min(max_speed, dist * kp_pos)
            
            error_world = target_pos - env.pos[0]
            if dist > 0.02:
                v_world = target_speed * (error_world / dist)
            else:
                v_world = np.zeros(3)

            # 2. Rotate world velocity into body-frame coordinates
            yaw = env.rpy[0, 2]
            vx_body = v_world[0] * np.cos(yaw) + v_world[1] * np.sin(yaw)
            vy_body = -v_world[0] * np.sin(yaw) + v_world[1] * np.cos(yaw)
            vz_body = v_world[2]

            # 3. Dynamic yaw control to point toward next waypoint
            yaw_rate = 0.0
            dist_xy = np.linalg.norm(target_pos[:2] - env.pos[0, :2])
            if dist_xy > 0.3:
                target_yaw = np.arctan2(target_pos[1] - env.pos[0, 1], target_pos[0] - env.pos[0, 0])
                yaw_err = target_yaw - yaw
                yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
                yaw_rate = np.clip(2.0 * yaw_err, -1.2, 1.2)

            vehicle.set_velocity(vx_body, vy_body, vz_body, yaw_rate=yaw_rate)

        elif state == "HOVER":
            # Hold zero velocity
            vehicle.set_velocity(0.0, 0.0, 0.0, yaw_rate=0.0)

        # Update controller and get motor RPMs
        rpm = vehicle.update(control_timestep=env.CTRL_TIMESTEP)

        # Step simulation environment
        obs, reward, terminated, truncated, info = env.step(rpm)

        # Push frame to background camera thread snap
        if env.VISION_ATTR:
            if not _cam_frame_queue.full():
                data_copy = mujoco.MjData(env.model)
                mujoco.mj_copyData(data_copy, env.model, env.data)
                _cam_frame_queue.put_nowait(data_copy)

        # Display latest front camera frame in Pygame
        if pygame_ok:
            with _latest_frame["lock"]:
                frame_rgb = _latest_frame["rgb"]
            if frame_rgb is not None:
                surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                surf_scaled = pygame.transform.scale(surf, (cam_width * display_scale, cam_height * display_scale))
                screen.blit(surf_scaled, (0, 0))
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Render at most 30 FPS to reduce rendering overhead and lag under WSLg
        current_time = time.time()
        if USE_GUI and current_time - last_render_time >= 1.0 / 30.0:
            env.render()
            last_render_time = current_time
            render_count += 1

        # Periodic telemetry log prints
        if step == 0:
            print(f"t=  0.0s | pos=[{env.pos[0,0]:+.2f}, {env.pos[0,1]:+.2f}, {env.pos[0,2]:+.2f}] | target_err={dist:.4f}")
        elif step % 48 == 0:
            now = time.time()
            elapsed_fps = now - last_fps_time
            current_loop_rate = loop_step_count / elapsed_fps
            current_render_fps = render_count / elapsed_fps
            
            loop_step_count = 0
            render_count = 0
            last_fps_time = now
            
            fps_str = f" | Loop Rate={current_loop_rate:.1f}Hz"
            if USE_GUI:
                fps_str += f" | Render FPS={current_render_fps:.1f}"
            print(f"t={step * env.CTRL_TIMESTEP:5.1f}s | mode={vehicle.mode:10s} | pos=[{env.pos[0,0]:+.2f}, {env.pos[0,1]:+.2f}, {env.pos[0,2]:+.2f}] | target_err={dist:.4f}{fps_str}")

        step += 1

        # Reset if crashed
        rpy = env.rpy[0]
        has_crashed = env.pos[0, 2] < 0.05 or env.pos[0, 2] > 5.0 or abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2
        if has_crashed:
            print("[MISSION] CRASH/OUT-OF-BOUNDS detected! Resetting mission...")
            obs, info = env.reset()
            vehicle.disarm()
            vehicle.arm()
            wp_idx = 0
            target_pos = waypoints[wp_idx][0]
            wp_name = waypoints[wp_idx][1]
            state = "TAKEOFF"
            vehicle.simple_takeoff(target_pos[2])

        # Pace loop to real-time
        elapsed = time.time() - start_time
        if elapsed < env.CTRL_TIMESTEP:
            time.sleep(env.CTRL_TIMESTEP - elapsed)

    # Clean up and exit
    _cam_frame_queue.put(None)
    cam_thread.join(timeout=3.0)
    env.close()

    # Save recorded frames
    if len(frames) > 0:
        os.makedirs("/home/firza/MuJoCo-drones-gym/multi_drone_mujoco/results", exist_ok=True)
        gif_path = "/home/firza/MuJoCo-drones-gym/multi_drone_mujoco/results/front_cam_yolo.gif"
        print(f"\nSaving front camera recording to {gif_path}...")
        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=40,
                loop=0
            )
            print("Successfully saved GIF recording!")
        except Exception as e:
            print(f"Could not save recording as GIF: {e}")

        # Also try to save as MP4 if imageio is available
        video_path = "/home/firza/MuJoCo-drones-gym/multi_drone_mujoco/results/front_cam_yolo.mp4"
        try:
            import imageio
            print(f"Attempting to save MP4 recording to {video_path}...")
            imageio.mimsave(video_path, frames, fps=24)
            print("Successfully saved MP4 recording!")
        except Exception as e:
            print(f"Could not save MP4 recording: {e}")

    print("Mission finished successfully.")

if __name__ == "__main__":
    main()
