import time
import jax
import os
# Force JAX to allocate memory dynamically as needed instead of pre-claiming 75%
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Limit compiler parallel threads to reduce peak host-RAM spikes during optimization
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
jax.config.update("jax_compilation_cache_dir", "/home/firza/MuJoCo-drones-gym/.jax_cache")

import jax.numpy as jnp
import numpy as np
import mediapy as media
import mujoco
from mujoco import mjx
import mujoco.viewer



from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax

def verify_env_fast():
    print("Initializing environment...")
    env = KRTIAviaryJax()
    print("dt:", env.dt, "sim_freq:", env.sim_freq, "sim_steps_per_ctrl:", env.sim_steps_per_ctrl)
    print("Simulated body mass:", env.mj_model.body_mass[env.drone_body_id])
    print("Controller's assumed mass:", env.mass)

    # Pre-compile actions for the 384 steps to prevent loop branching overhead
    # 48 steps per phase (~1s at 48Hz): hover, forward, backward, right, left, up, down, hover
    actions = jnp.zeros((384, 4))
    actions = actions.at[48:96, 0].set(1.0)    # Move forward
    actions = actions.at[96:144, 0].set(-1.0)  # Move backward
    actions = actions.at[144:192, 1].set(1.0)  # Move right (positive action[1] is right)
    actions = actions.at[192:240, 1].set(-1.0) # Move left
    actions = actions.at[240:288, 2].set(-1.0) # Move up (negative action[2] is up)
    actions = actions.at[288:336, 2].set(1.0)  # Move down
    
    # Run the entire simulation loop inside JAX using lax.scan for instant execution
    def sim_step(carry_state, action):
        next_state = env.step(carry_state, action)
        # Extract only what the renderer needs to minimize device-to-host footprint
        render_data = {
            "qpos": next_state.pipeline_state.qpos,
            "qvel": next_state.pipeline_state.qvel
        }
        return next_state, render_data

    @jax.jit
    def run_simulation(init_state, action_seq):
        return jax.lax.scan(sim_step, init_state, action_seq)

    print("Resetting environment...")
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    print("JIT compiling simulation function...")
    start_time = time.time()
    final_state, trajectory = run_simulation(state, actions)
    trajectory["qpos"].block_until_ready()
    print(f"Simulation completed on device in {time.time() - start_time:.2f} seconds!")

    # Bring data back to CPU IN A SINGLE BATCH transfer
    print("Transferring trajectory to CPU...")
    qpos_history = np.array(trajectory["qpos"])
    qvel_history = np.array(trajectory["qvel"])
    
    # Setup MuJoCo Renderer
    renderer = mujoco.Renderer(env.mj_model, height=240, width=320)
    mj_data = mujoco.MjData(env.mj_model)
    
    import cv2
    from multi_drone_mujoco.examples.krti_arena import get_gate_corners, project_point
    
    # Cache camera details
    cam_name = "drone0_cam"
    cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = env.mj_model.cam_fovy[cam_id]
    
    def compute_fake_yolo_local(mj_data):
        gate_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")
        if gate_body_id < 0:
            return None
        cam_pos = mj_data.cam_xpos[cam_id].copy()
        cam_mat = mj_data.cam_xmat[cam_id].copy()
        T_gate = mj_data.xpos[gate_body_id].copy()
        R_gate = mj_data.xmat[gate_body_id].copy().reshape(3, 3)
        local_corners = get_gate_corners("single")
        px_list = []
        for pt_local in local_corners:
            pt_world = R_gate @ pt_local + T_gate
            px = project_point(pt_world, cam_pos, cam_mat, fovy, 320, 240)
            if px is not None:
                px_x, px_y = px
                if 0 <= px_x <= 320 and 0 <= px_y <= 240:
                    px_list.append(px)
        if len(px_list) < 2:
            return None
        xs = [p[0] for p in px_list]
        ys = [p[1] for p in px_list]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    frames = []
    print("Rendering FPV video frames and updating OpenCV GUI...")
    # Render every 2nd frame on the CPU using the pre-compiled history
    for i in range(0, 384, 2):
        mj_data.qpos[:] = qpos_history[i]
        mj_data.qvel[:] = qvel_history[i]
        mujoco.mj_forward(env.mj_model, mj_data)
        
        # Render the FPV camera view
        renderer.update_scene(mj_data, camera=cam_name)
        img = renderer.render()
        
        # Convert RGB to BGR for OpenCV
        hud_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw camera center crosshair
        cx, cy = 320 // 2, 240 // 2
        cv2.drawMarker(hud_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=1)
        
        # Compute and draw YOLO bounding box
        box = compute_fake_yolo_local(mj_data)
        if box is not None:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(hud_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw line to target center
            tx = (x_min + x_max) // 2
            ty = (y_min + y_max) // 2
            cv2.line(hud_img, (cx, cy), (tx, ty), (0, 0, 255), 1)
            cv2.circle(hud_img, (tx, ty), 4, (0, 0, 255), -1)
            
            # Label
            cv2.putText(hud_img, "GATE LOCKED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(hud_img, "TARGET LOST (SEARCHING...)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
        # Display the live window
        cv2.imshow("Drone FPV View (JAX Verification)", hud_img)
        cv2.waitKey(40)  # ~24 FPS playback speed
        
        # Append RGB image (with overlays) to video frames list
        frames.append(cv2.cvtColor(hud_img, cv2.COLOR_BGR2RGB))
        
    print("Simulation rendering finished. Closing OpenCV windows...")
    cv2.destroyAllWindows()
        
    video_path = "/home/firza/MuJoCo-drones-gym/jax_takeoff_verification.mp4"
    print(f"Saving FPV video to {video_path}...")
    media.write_video(video_path, frames, fps=24)
    print(f"Video saved to {video_path}!")

if __name__ == "__main__":
    verify_env_fast()
