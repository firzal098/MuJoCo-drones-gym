import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2


def interpolate_series(history, n_points=11):
    """
    Resample a (step, value) history onto n_points evenly spaced
    across normalized progress t=0..1, using linear interpolation.
    Returns list of (t, step_equiv, value).
    """
    if len(history) < 2:
        step, val = history[0]
        return [(0.0, step, val)]

    steps = np.array([s for s, _ in history], dtype=float)
    vals = np.array([v for _, v in history], dtype=float)

    t_grid = np.linspace(0.0, 1.0, n_points)
    step_grid = steps[0] + t_grid * (steps[-1] - steps[0])
    val_interp = np.interp(step_grid, steps, vals)

    return list(zip(t_grid, step_grid, val_interp))


def trend_summary(history):
    vals = [v for _, v in history]
    n = len(vals)
    if n < 3:
        return "not enough data"
    third = max(n // 3, 1)
    early = sum(vals[:third]) / third
    mid = sum(vals[third:2 * third]) / max(len(vals[third:2 * third]), 1)
    late = sum(vals[-third:]) / third

    def arrow(a, b):
        if b > a * 1.05:
            return "up"
        if b < a * 0.95:
            return "down"
        return "flat"

    return f"early→mid: {arrow(early, mid)}, mid→late: {arrow(mid, late)}"


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

    tags_of_interest = [
        tag for tag in data.keys()
        if "eval/" in tag or tag in ["crashed", "cleared_gate", "gate_collided"]
    ]

    for tag in tags_of_interest:
        history = sorted(data[tag])
        if not history:
            continue

        points = interpolate_series(history, n_points=11)  # t = 0, 0.1, ..., 1.0

        print(f"\nTag: {tag}")
        print(f"  {'t':>5} | {'step':>10} | {'value':>10}")
        for t, step, val in points:
            print(f"  {t:5.1f} | {step:10.0f} | {val:10.4f}")

        print(f"  shape: {trend_summary(history)}")


if __name__ == "__main__":
    inspect_latest_run_fast()