"""Quick sanity check that the container has core + fork dependencies."""

import importlib
import sys


MODULES = [
    "gymnasium",
    "mujoco",
    "mujoco.mjx",
    "numpy",
    "stable_baselines3",
    "pettingzoo",
    "matplotlib",
    "PIL",
    "jax",
    "brax",
    "flax",
    "tensorboard",
    "torch",
    "mediapy",
    "cv2",
    "onnxruntime",
    "mujoco_warp",
    "multi_drone_mujoco",
]


def main() -> int:
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "ok")
            print(f"  OK  {name:24s} {version}")
        except Exception as exc:
            print(f"FAIL  {name:24s} {exc}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} module(s) failed to import.", file=sys.stderr)
        return 1

    print("\nAll dependencies imported successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
