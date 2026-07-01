import time
import numpy as np
from multi_drone_mujoco.examples.krti_arena import KRTIAviary
from multi_drone_mujoco.utils.enums import DroneModel

def verify_env_gui():
    print("Initializing environment with GUI...")
    # gui=True launches the interactive passive viewer window
    # obstacles=True spawns the KRTI arena waypoints and gates
    env = KRTIAviary(gui=True, obstacles=True)
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print("Stepping environment (Hover and Takeoff simulation)...")
    print("You should see the MuJoCo GUI open. You can interact with the 3D window.")
    
    # Run for 240 steps (5 seconds at 48Hz)
    for i in range(240):
        # Stepping action: hover for 1s, takeoff for 2s, hover for 2s
        if i < 48:
            # First second: Hover in place at spawn height (z=0.25m)
            action = np.array([0.0, 0.0, 0.0, 0.0])
        elif i < 144:
            # Seconds 1 to 3: Command negative Z action to take off higher
            action = np.array([0.0, 0.0, -1.0, 0.0])
        else:
            # Seconds 3 to 5: Hover in the new higher position
            action = np.array([0.0, 0.0, 0.0, 0.0])
            
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if i % 48 == 0:
            # Get drone position from environment data
            drone_pos = env.pos[0]
            print(f"Time {i//48}s - Action: {action.tolist()} - Z-Pos: {drone_pos[2]:.2f}m")
            
        # Slow down step rate to match real-time
        time.sleep(1.0 / 48.0)
        
        if terminated or truncated:
            break
            
    print("Simulation finished. Keeping GUI viewer open. Press Ctrl+C in the terminal or close the window to exit...")
    try:
        while env._viewer is not None and env._viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
        
    print("Closing viewer...")
    env.close()
    print("Done!")

if __name__ == "__main__":
    verify_env_gui()
