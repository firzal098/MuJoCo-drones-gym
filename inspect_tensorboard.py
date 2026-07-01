import glob
import os

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2

def inspect_latest_run_fast():
    tb_dir = "/home/firza/MuJoCo-drones-gym/results/krti_single_rl_jax/tensorboard"
    runs = sorted(
        glob.glob(os.path.join(tb_dir, "run_*")), key=lambda x: int(x.split("_")[-1])
    )
    if not runs:
        print("No runs found.")
        return

    latest_run = runs[-1]
    print(f"Inspecting latest run: {latest_run}")

    event_files = glob.glob(os.path.join(latest_run, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found in run.")
        return

    latest_event_file = max(event_files, key=os.path.getmtime)
    print(f"Reading {latest_event_file}")

    loader = RawEventFileLoader(latest_event_file)

    # Store recent values for tags
    data = {}
    for event_bytes in loader.Load():
        event = event_pb2.Event.FromString(event_bytes)
        if not event.HasField("summary"):
            continue
        for value in event.summary.value:
            tag = value.tag
            if tag not in data:
                data[tag] = []
            if value.HasField("simple_value"):
                data[tag].append((event.step, value.simple_value))
            elif value.HasField("tensor"):
                import tensorflow as tf

                val = tf.make_ndarray(value.tensor)
                data[tag].append((event.step, float(val)))

    # Print tags of interest
    tags_of_interest = [
        tag
        for tag in data.keys()
        if "eval/" in tag or tag in ["crashed", "cleared_gate", "gate_collided"]
    ]
    for tag in tags_of_interest:
        history = data[tag]
        if not history:
            continue
        print(f"\nTag: {tag}")
        # Show last 5 entries
        for step, val in history[-5:]:
            print(f"  Step: {step}, Value: {val:.4f}")

        for step, val in history[-5:]:
            print(f"  Step: {step}, Value: {val:.4f}")

if __name__ == "__main__":
    inspect_latest_run_fast()
