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

# TOGGLES FOR TESTING MODES:
# 1. USE_HEURISTIC_TEST_FLIGHT: Fly towards and through the gate using a proportional controller.
# 2. USE_LOOK_AWAY_TEST_FLIGHT: Spin 180 degrees away from the gate to test if the bounding box cleanly disappears.
# 3. If both are False, the script loads and flies with the trained JAX PPO neural network policy.
# Reset to False by default to prevent overlapping diagnostic overrides.
USE_HEURISTIC_TEST_FLIGHT = False
USE_LOOK_AWAY_TEST_FLIGHT = False

def main():
    env = KRTIAviaryJax()
    
    predict_fn = None
    # Initialize neural network inference if we are running the actual learned policy
    if not USE_HEURISTIC_TEST_FLIGHT and not USE_LOOK_AWAY_TEST_FLIGHT:
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size=16,  # Matches the updated 16-dimensional observation size
            action_size=4,
        )
        inference_fn = ppo_networks.make_inference_fn(ppo_network)
        
        # Load the parameters
        model_path = "./results/krti_single_rl_jax/finalised/final"
        if not os.path.exists(model_path):
            print(f"[ERROR] Policy file not found at {model_path}. Please train a model or enable diagnostic modes.")
            return
            
        print(f"Loading JAX policy from {model_path}...")
        params = model.load_params(model_path)
        predict_fn = jax.jit(inference_fn(params, deterministic=True)) 
    elif USE_LOOK_AWAY_TEST_FLIGHT:
        print("[LOOK-AWAY MODE] Turning nose 180 degrees away from gate to verify bounding box occlusion.")
    else:
        print("[HEURISTIC MODE] Flying via closed-loop Proportional Controller to verify gate bounding box math.")

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
            
            # Keep point if in front of the lens (depth > 0)
            if z_c < 0:
                depth = -z_c
                f_y = 1.0 / np.tan(np.deg2rad(fovy) / 2.0)
                f_x = f_y * (240.0 / 320.0)  # height / width

                ndc_x = f_x * (x_c / depth)
                ndc_y = f_y * (y_c / depth)

                px_x = (ndc_x + 1.0) / 2.0 * 320
                px_y = (1.0 - ndc_y) / 2.0 * 240
                px_list.append((px_x, px_y))
                
        if not px_list:
            return None
            
        xs = [p[0] for p in px_list]
        ys = [p[1] for p in px_list]
        
        # Verify if the unclipped bounding box has any physical overlap with the viewport bounds
        if min(xs) > 320 or max(xs) < 0 or min(ys) > 240 or max(ys) < 0:
            return None

        # Clip final bounding box limits to screen dimensions
        x_min = int(np.clip(min(xs), 0, 320))
        y_min = int(np.clip(min(ys), 0, 240))
        x_max = int(np.clip(max(xs), 0, 320))
        y_max = int(np.clip(max(ys), 0, 240))
        
        # Reject collapsed bounding boxes
        if x_max - x_min < 2 or y_max - y_min < 2:
            return None
            
        return x_min, y_min, x_max, y_max

    frames = []
    episode_reward = 0.0
    steps = 0
    
    print("Starting flight evaluation... Close OpenCV window or press Esc to exit.")
    try:
        step_fn = jax.jit(env.step)
        
        while steps < 600:
            rng, action_rng = jax.random.split(rng)
            
            if USE_LOOK_AWAY_TEST_FLIGHT:
                # ------------------------------------------------------------
                # DIAGNOSTIC LOOK-AWAY CLOSED-LOOP CONTROLLER
                # ------------------------------------------------------------
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
                # Corrected proportional sign mapping to match the FRD body z command
                act_vertical = -float(np.clip(0.7 * rel_gate_body[2], -0.3, 0.3))
                
                # Spin yaw exactly 180 degrees away from the target gate position
                yaw_error_away = np.arctan2(-rel_gate_body[1], -rel_gate_body[0])
                act_yaw = float(np.clip(1.3 * yaw_error_away, -0.4, 0.4))
                
                action = jnp.array([act_forward, act_lateral, act_vertical, act_yaw])

            elif USE_HEURISTIC_TEST_FLIGHT:
                # ------------------------------------------------------------
                # HEURISTIC CLOSED-LOOP PROPORTIONAL CONTROLLER (FORWARD)
                # ------------------------------------------------------------
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
                # Corrected proportional sign mapping to prevent driving into the ground
                act_vertical = -float(np.clip(0.7 * rel_gate_body[2], -0.3, 0.3))
                
                # Steer nose toward gate center
                yaw_error = np.arctan2(rel_gate_body[1], rel_gate_body[0])
                act_yaw = float(np.clip(1.3 * yaw_error, -0.4, 0.4))
                
                action = jnp.array([act_forward, act_lateral, act_vertical, act_yaw])
            else:
                # Evaluate network action from current 16-dimensional observation vector
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
            
            # Calculate distance to display on HUD
            cur_drone_pos = np.array(state.pipeline_state.qpos[0:3])
            cur_gate_pos = np.array(state.info["gate_pos"])
            dist_gate = np.linalg.norm(cur_gate_pos - cur_drone_pos)

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
                
            # Render distance to gate metrics on overlay
            cv2.putText(hud_img, f"GATE DIST: {dist_gate:.2f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
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