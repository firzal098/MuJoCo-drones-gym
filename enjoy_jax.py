import os
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

def main():
    env = KRTIAviaryJax()
    
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=13,
        action_size=4,
    )
    inference_fn = ppo_networks.make_inference_fn(ppo_network)
    
    # Load the parameters
    model_path = "./results/krti_single_rl_jax/finalised/final"
    if not os.path.exists(model_path):
        print(f"[ERROR] Policy file not found at {model_path}.")
        return
        
    print(f"Loading JAX policy from {model_path}...")
    params = model.load_params(model_path)
    
    # 3. SET DETERMINISTIC TO TRUE HERE
    # This strips the exploration noise while maintaining the internal function shape
    predict_fn = jax.jit(inference_fn(params, deterministic=True)) 
    
    print("Resetting environment...")
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    
    # Setup rendering
    renderer = mujoco.Renderer(env.mj_model, height=240, width=320)
    mj_data = mujoco.MjData(env.mj_model)
    cam_name = "drone0_cam"
    cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = env.mj_model.cam_fovy[cam_id]
    
    from multi_drone_mujoco.examples.krti_arena import get_gate_corners, project_point
    
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
    episode_reward = 0.0
    steps = 0
    
    print("Starting JAX policy flight evaluation... Close OpenCV window or press Esc to exit.")
    try:
        step_fn = jax.jit(env.step)
        
        while steps < 600:
            rng, action_rng = jax.random.split(rng)
            
            # 4. KEEP THE action_rng ARGUMENT HERE SO THE SIGNATURE COMPILES PERFECTLY
            action, _ = predict_fn(state.obs, action_rng)
            
            # Step the environment
            state = step_fn(state, action)
            episode_reward += float(state.reward)
            steps += 1
            
            # Sync JAX state back to MuJoCo CPU structures for rendering
            mj_data.qpos[:] = np.array(state.pipeline_state.qpos)
            mj_data.qvel[:] = np.array(state.pipeline_state.qvel)
            
            # Sync randomized gate position from JAX state info to rendering model
            gate_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a")
            if gate_body_id >= 0:
                gate_pos = np.array(state.info["gate_pos"])
                gate_body_pos = gate_pos.copy()
                gate_body_pos[0] -= 0.95156
                gate_body_pos[2] = 0.0
                env.mj_model.body_pos[gate_body_id] = gate_body_pos
                
            mujoco.mj_forward(env.mj_model, mj_data)
            
            # Render FPV camera frame
            renderer.update_scene(mj_data, camera=cam_name)
            img = renderer.render()
            
            # Convert RGB to BGR for OpenCV overlay draw
            hud_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cx, cy = 320 // 2, 240 // 2
            cv2.drawMarker(hud_img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=1)
            
            box = compute_fake_yolo_local(mj_data)
            if box is not None:
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(hud_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                tx, ty = (x_min + x_max) // 2, (y_min + y_max) // 2
                cv2.line(hud_img, (cx, cy), (tx, ty), (0, 0, 255), 1)
                cv2.circle(hud_img, (tx, ty), 4, (0, 0, 255), -1)
                cv2.putText(hud_img, f"LOCK: {steps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(hud_img, "SEARCHING...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
            cv2.imshow("Drone JAX Policy Evaluation", hud_img)
            if cv2.waitKey(30) & 0xFF == 27: # Esc
                break
                
            frames.append(cv2.cvtColor(hud_img, cv2.COLOR_BGR2RGB))
            
            if float(state.done) > 0.5:
                print(f"Episode finished! Steps: {steps}, Total Reward: {episode_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        cv2.destroyAllWindows()
        
    if frames:
        video_path = "/home/firza/MuJoCo-drones-gym/enjoy_jax_flight.mp4"
        print(f"Saving flight video to {video_path}...")
        media.write_video(video_path, frames, fps=24)
        print(f"Video saved to {video_path}!")

if __name__ == "__main__":
    main()