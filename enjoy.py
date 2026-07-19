import os
import glob
import argparse
import subprocess
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import PPO

# Import custom aviary/arena components
from train_single_rl import SingleGateTrainingWrapper, KRTIAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType


def get_latest_checkpoint(folder="./results/"):
    """Finds the most recently modified PPO checkpoint zip file in results/."""
    zip_files = glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True)
    if not zip_files:
        return None
    zip_files.sort(key=os.path.getmtime, reverse=True)
    return zip_files[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize trained RL model for KRTI drone gate navigation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the model checkpoint (.zip) to evaluate. If omitted, uses the latest checkpoint in results/."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive 3D GUI window and render/save evaluation runs directly to an MP4 video."
    )
    parser.add_argument(
        "--iterations", "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes to run (default: 3)."
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="./results/evaluation.mp4",
        help="Destination path for MP4 video when --no-gui is used (default: ./results/evaluation.mp4)."
    )
    return parser.parse_args()


def save_mp4(frames, filename, fps=30):
    """Saves a sequence of RGB numpy frames as an H.264 MP4 video file compatible with IDE/web players."""
    if not frames:
        return
    height, width, _ = frames[0].shape
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tmp_filename = filename + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_filename, fourcc, fps, (width, height))
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()

    # Convert to web/IDE-compatible H.264 codec using ffmpeg
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_filename,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            filename
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        print(f"\n[VIDEO SAVED] Successfully saved H.264 MP4 video to: {os.path.abspath(filename)}\n")
    except Exception:
        if os.path.exists(tmp_filename):
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(tmp_filename, filename)
        print(f"\n[VIDEO SAVED] Saved MP4 video to: {os.path.abspath(filename)}\n")


def enjoy():
    args = parse_args()

    # Determine model path
    if args.model:
        model_path = args.model
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
        if not os.path.exists(model_path + ".zip"):
            print(f"\n[ERROR] Model checkpoint not found at: {model_path}.zip\n")
            return
        model_file = model_path + ".zip"
    else:
        model_file = get_latest_checkpoint("./results/")
        if model_file is None:
            print("\n[ERROR] No saved checkpoints (.zip files) found in ./results/\n")
            return

    gui_enabled = not args.no_gui

    print("=" * 60)
    print("Launching KRTI Drone RL Model Evaluation")
    print(f"  Model Checkpoint : {os.path.abspath(model_file)}")
    print(f"  Mode             : {'Interactive 3D GUI' if gui_enabled else 'Headless MP4 Recording'}")
    print(f"  Iterations       : {args.iterations}")
    if not gui_enabled:
        print(f"  Video Output     : {os.path.abspath(args.output_video)}")
    print("=" * 60)

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
        initial_xyzs=np.array([[0.92, 24.47, 1.0]]),
        initial_rpys=np.array([[0.0, 0.0, -np.pi/2]])
    )

    env = SingleGateTrainingWrapper(base_env)
    model = PPO.load(model_file, env=env)

    video_frames = []

    try:
        for ep in range(1, args.iterations + 1):
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0

            print(f"\n--- Episode {ep}/{args.iterations} Started ---")

            if not gui_enabled:
                video_frames.append(env.capture_frame())

            while True:
                import time
                start_time = time.time()

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                if not gui_enabled:
                    video_frames.append(env.capture_frame())
                else:
                    elapsed = time.time() - start_time
                    sleep_time = (1.0 / 40.0) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if terminated or truncated:
                    metrics = info.get("metrics", {})
                    outcome = "CLEARED GATE" if metrics.get("cleared_gate", 0) else "CRASHED"
                    dist_rem = metrics.get("gate_distance", 0.0)
                    print(f"Episode {ep} Finished [{outcome}] | Dist Remaining: {dist_rem:.2f}m | Reward: {episode_reward:.2f} | Steps: {step_count}")
                    break

        if not gui_enabled and video_frames:
            save_mp4(video_frames, args.output_video)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation loop stopped by user.")
        if not gui_enabled and video_frames:
            save_mp4(video_frames, args.output_video)
    finally:
        env.close()


if __name__ == "__main__":
    enjoy()
