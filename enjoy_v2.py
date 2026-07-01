import os
import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Import custom aviary/arena components
from train_single_rl_v2 import SingleGateTrainingWrapper
from multi_drone_mujoco.examples.krti_arena import KRTIAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType

def get_latest_checkpoint(folder):
    """Finds the most recently modified PPO checkpoint zip file in the folder."""
    zip_files = glob.glob(os.path.join(folder, "*.zip"))
    if not zip_files:
        return None
    # Sort by modification time
    zip_files.sort(key=os.path.getmtime, reverse=True)
    return zip_files[0]

def enjoy_v2():
    checkpoint_dir = "./results/krti_single_rl_v2/"
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        print(f"\n[ERROR] No saved checkpoints (.zip files) found in '{checkpoint_dir}' yet.")
        print("Please wait for training to save at least one checkpoint (every 15,000 steps).\n")
        return
        
    print("=" * 60)
    print(f"Loading latest checkpoint: {os.path.abspath(latest_checkpoint)}")
    print("Spawning drone visualization... Close window or press Ctrl+C to exit.")
    print("=" * 60)
    
    gui_enabled = True          # Set to True to show the 3D MuJoCo Viewer
    fpv_cam_enabled = False       # Set to True to show the OpenCV YOLO HUD

    # Initialize environment
    base_env = KRTIAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        gui=gui_enabled,
        vision_attributes=True,
        render_mode="human" if gui_enabled else None,
        initial_xyzs=np.array([[0.92, 24.47, 0.25]]),
        initial_rpys=np.array([[0.0, 0.0, -np.pi/2]])
    )
    
    # Wrap environment (from v2)
    env = SingleGateTrainingWrapper(base_env)
    
    # Configure FPV HUD window
    env.show_fpv_gui = fpv_cam_enabled
    
    # Load model
    model = PPO.load(latest_checkpoint, env=env)
    
    try:
        obs, info = env.reset()
        episode_reward = 0.0
        
        import time
        while True:
            start_time = time.time()
            
            # Predict action deterministically
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Sleep to match real-time (control frequency is 48 Hz)
            elapsed = time.time() - start_time
            sleep_time = (1.0 / 40.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if terminated or truncated:
                print(f"Episode finished! Total Reward: {episode_reward:.2f}")
                print("Resetting environment for next run...\n")
                obs, info = env.reset()
                episode_reward = 0.0
                
    except KeyboardInterrupt:
        print("\nExiting visualization.")
    finally:
        env.close()

if __name__ == "__main__":
    enjoy_v2()
