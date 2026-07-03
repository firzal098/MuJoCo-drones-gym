"""
train_single_rl_sbx_mjx.py
===========================
MJX-physics + SBX-PPO training for single-gate drone navigation.

Simulation backend : MuJoCo MJX (GPU / XLA) — jax.vmap(mjx.step)
RL backend         : Stable-Baselines Jax (SBX) PPO

Compared with train_single_rl_sbx.py:
  • Physics moves from 16 CPU subprocesses (mj_step) to a single GPU kernel
    running NUM_ENVS environments in parallel via jax.vmap(mjx.step).
  • Observation space is 20-dim (KRTIAviaryJax) instead of 100-dim.
    Saved SBX checkpoints from train_single_rl_sbx.py are NOT compatible.
  • Auto-reset is handled inside the JIT'd step via jnp.where tree-merge —
    no Python-level branching per step.

Install / verify GPU before running:
    pip install -U "jax[cuda12]"          # for CUDA 12.x
    python -c "import jax; print(jax.devices())"
"""

# ── JAX / XLA env vars — must be set BEFORE importing jax ────────────────────
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=1")
# Higher fraction than train_single_rl_sbx.py: MJX physics also lives on GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import sys
import time
import numpy as np

# ── JAX compilation cache (warm restarts are significantly faster) ────────────
import jax
import jax.numpy as jnp

_jax_cache = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"),
)
os.makedirs(_jax_cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache)

from gymnasium import spaces

# ── SBX ───────────────────────────────────────────────────────────────────────
try:
    from sbx import PPO
except ImportError as e:
    raise ImportError(
        "SBX is not installed. Please run:\n"
        "  pip install sbx-rl\n"
        "or:\n"
        "  pip install git+https://github.com/araffin/sbx"
    ) from e

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ── MJX environment ───────────────────────────────────────────────────────────
from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax


# ══════════════════════════════════════════════════════════════════════════════
# Global configuration
# ══════════════════════════════════════════════════════════════════════════════

# Parallel environments — all run in one GPU vmap kernel.
# Tune down (e.g. 256 or 512) if you hit VRAM limits.
NUM_ENVS = 1024

CURRICULUM_STAGES = [
    {"level": 1, "steps": 3_000_000, "lr": 3.0e-4},  # Fixed config, low speed
    {"level": 2, "steps": 3_000_000, "lr": 2.5e-4},  # Minor variations
    {"level": 3, "steps": 4_000_000, "lr": 2.0e-4},  # Moderate variations
    {"level": 4, "steps": 4_000_000, "lr": 1.5e-4},  # Camera noise, aggressive offset
    {"level": 5, "steps": 6_000_000, "lr": 1.0e-4},  # Full domain randomisation
]


# ══════════════════════════════════════════════════════════════════════════════
# MJXVecEnv — bridges KRTIAviaryJax (Brax / MJX) to the SB3 VecEnv API
# ══════════════════════════════════════════════════════════════════════════════
class MJXVecEnv(VecEnv):
    """
    Vectorized MJX environment compatible with the SB3 / SBX VecEnv API.

    All ``num_envs`` environments run in a single JIT-compiled GPU kernel via
    ``jax.vmap``.  Auto-reset is handled inside the JIT function with
    ``jnp.where`` over the full Brax State pytree — no Python branching in
    the hot path.

    Episode statistics (return, length, reward components) are accumulated in
    numpy arrays and exposed through ``info["episode"]`` / ``info["ep_metrics"]``
    at episode boundaries, matching the SB3 Monitor / TensorBoard conventions.

    Parameters
    ----------
    env : KRTIAviaryJax
        A freshly created (un-reset) MJX environment instance.
    num_envs : int
        Number of parallel environments to simulate on GPU.
    seed : int
        Base PRNG seed.
    """

    # Keys present in KRTIAviaryJax.step() -> state.metrics
    METRIC_KEYS = (
        "crashed",
        "cleared_gate",
        "gate_collided",
        "gate_distance",
        "reward_progress",
        "reward_centering",
        "reward_speed",
        "reward_attitude",
        "reward_smooth",
        "reward_terminal",
    )

    def __init__(self, env, num_envs, seed=42):
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.observation_size,),
            dtype=np.float32,
        )
        act_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.action_size,),
            dtype=np.float32,
        )
        super().__init__(num_envs, obs_space, act_space)

        self.env = env
        self._key = jax.random.PRNGKey(seed)
        self._states = None
        self._pending_actions = None
        self._start_time = time.time()

        # Per-env episode accumulators (numpy — cheap Python-side bookkeeping)
        self._ep_returns = np.zeros(num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self._ep_metrics = {
            k: np.zeros(num_envs, dtype=np.float32) for k in self.METRIC_KEYS
        }

        # ── Build JIT-compiled functions via explicit closures ─────────────────
        # Capture env as _env so JAX never re-traces due to self mutation.
        _env = env

        @jax.jit
        def _jit_reset(keys):
            """vmap reset across all envs."""
            return jax.vmap(_env.reset)(keys)

        @jax.jit
        def _jit_step_reset(states, actions, reset_keys):
            """
            One combined GPU kernel per SBX step:

            1. Step all envs.
            2. Reset all envs (outputs discarded for non-done envs).
            3. Merge via jnp.where — done envs use reset_state.

            Returns
            -------
            merged_states   : new state to store (reset obs for done envs)
            terminal_obs    : obs from next_states *before* reset (for info)
            rewards         : step reward (num_envs,)
            done_flags      : float32 done (num_envs,)
            metrics         : dict of per-env metric arrays
            """
            next_states  = jax.vmap(_env.step)(states, actions)
            reset_states = jax.vmap(_env.reset)(reset_keys)

            done = next_states.done.astype(bool)  # shape: (num_envs,)

            def _where(nxt, rst):
                """
                Select rst[i] where done[i] else nxt[i].
                Handles arbitrary leaf shapes by broadcasting the done mask.
                Non-JAX leaves (Python scalars) are passed through unchanged.
                """
                if not hasattr(nxt, "dtype"):
                    # Not a JAX array — skip merging
                    return nxt
                if nxt.ndim == 0:
                    # 0-d arrays should not appear after vmap; keep as-is
                    return nxt
                shape = done.shape + (1,) * (nxt.ndim - 1)
                return jnp.where(done.reshape(shape), rst, nxt)

            merged = jax.tree_util.tree_map(_where, next_states, reset_states)

            return (
                merged,
                next_states.obs,      # terminal obs (before reset)
                next_states.reward,
                next_states.done,
                next_states.metrics,
            )

        self._jit_reset      = _jit_reset
        self._jit_step_reset = _jit_step_reset

        print(
            f"[MJXVecEnv] {num_envs} envs | obs={env.observation_size} | "
            f"act={env.action_size}\n"
            "  First reset/step triggers JIT compilation (~1-3 min)."
        )

    # ── Key management ─────────────────────────────────────────────────────────
    def _split_keys(self, n):
        """Split self._key into n fresh sub-keys (mutates self._key)."""
        self._key, *subkeys = jax.random.split(self._key, n + 1)
        return jnp.stack(subkeys)

    # ── VecEnv API ─────────────────────────────────────────────────────────────
    def reset(self):
        """Reset ALL environments and return initial observations (num_envs, obs_dim)."""
        keys = self._split_keys(self.num_envs)
        self._states = self._jit_reset(keys)

        # Reset episode accumulators
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        for k in self.METRIC_KEYS:
            self._ep_metrics[k][:] = 0.0
        self._start_time = time.time()

        return np.array(self._states.obs, dtype=np.float32)

    def step_async(self, actions):
        """Store actions for the upcoming step_wait() call."""
        self._pending_actions = jnp.array(actions, dtype=jnp.float32)

    def step_wait(self):
        """
        Execute one environment step on GPU and return (obs, rewards, dones, infos).

        For done environments:
            - obs[i]                          : first obs of the new episode
            - infos[i]["terminal_observation"]: final obs before done
            - infos[i]["episode"]             : {"r": ..., "l": ...}
            - infos[i]["ep_metrics"]          : reward component dict
        """
        assert self._states is not None, "Call reset() before step_wait()."
        assert self._pending_actions is not None, "Call step_async() before step_wait()."

        reset_keys = self._split_keys(self.num_envs)

        merged, term_obs_jax, rewards_jax, done_jax, metrics_jax = (
            self._jit_step_reset(self._states, self._pending_actions, reset_keys)
        )
        self._states          = merged
        self._pending_actions = None

        # Transfer from GPU to CPU once per step (at the VecEnv boundary)
        obs_np      = np.array(merged.obs,   dtype=np.float32)  # reset obs for done envs
        rewards_np  = np.array(rewards_jax,  dtype=np.float32)
        dones_np    = np.array(done_jax).astype(bool)
        term_obs_np = np.array(term_obs_jax, dtype=np.float32)  # pre-reset obs

        # Accumulate per-env episode statistics
        self._ep_returns += rewards_np
        self._ep_lengths += 1
        for k in self.METRIC_KEYS:
            self._ep_metrics[k] += np.array(metrics_jax[k], dtype=np.float32)

        # Build info dicts — only iterate over done envs for efficiency
        infos = [{} for _ in range(self.num_envs)]
        done_indices = np.where(dones_np)[0]
        now = time.time()
        for i in done_indices:
            infos[i]["episode"] = {
                "r": float(self._ep_returns[i]),
                "l": int(self._ep_lengths[i]),
                "t": now - self._start_time,
            }
            infos[i]["ep_metrics"] = {
                k: float(self._ep_metrics[k][i]) for k in self.METRIC_KEYS
            }
            infos[i]["terminal_observation"] = term_obs_np[i]
            # Reset per-env accumulators
            self._ep_returns[i] = 0.0
            self._ep_lengths[i] = 0
            for k in self.METRIC_KEYS:
                self._ep_metrics[k][i] = 0.0

        return obs_np, rewards_np, dones_np, infos

    def close(self):
        pass  # MJX holds no OS-level resources to release

    # ── Required abstract stubs (no meaningful semantics for a GPU-monolith env)
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        fn = getattr(self.env, method_name)
        return [fn(*method_args, **method_kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        return [seed] * self.num_envs


# ══════════════════════════════════════════════════════════════════════════════
# TensorBoard callback
# (same pattern as train_single_rl_sbx.py — bridges SBX -> torch SummaryWriter)
# ══════════════════════════════════════════════════════════════════════════════
class TBMetricsCallback(BaseCallback):
    """
    Forwards episode metrics to a torch SummaryWriter so that multi-stage
    global steps are tracked correctly across curriculum stages.
    """

    def __init__(self, writer, global_offset=0, verbose=0):
        super().__init__(verbose)
        self.writer        = writer
        self.global_offset = global_offset

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.writer is None:
            return
        global_step = self.global_offset + self.num_timesteps
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info:
                self.writer.add_scalar(
                    "curriculum/episode_reward", ep_info["r"], global_step
                )
                self.writer.add_scalar(
                    "curriculum/episode_length", ep_info["l"], global_step
                )
            ep_metrics = info.get("ep_metrics")
            if ep_metrics:
                for tag, value in ep_metrics.items():
                    self.writer.add_scalar(tag, float(value), global_step)


# ══════════════════════════════════════════════════════════════════════════════
# Training entry-point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  SBX PPO + MuJoCo MJX — KRTI Single-Gate Curriculum")
    all_devs = jax.devices()
    gpu_devs = [d for d in all_devs if "cuda" in str(d).lower() or "gpu" in str(d).lower()]
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  JAX devices : {all_devs}")
    if gpu_devs:
        print(f"  ✓ GPU: {gpu_devs[0]}  (MJX physics + SBX policy on GPU)")
    else:
        print(
            "  ⚠ No GPU detected — MJX will fall back to CPU.\n"
            "    Install 'jax[cuda12]' for GPU acceleration."
        )
    print(f"  Parallel envs : {NUM_ENVS}")
    print("=" * 60)

    output_dir     = "./results/krti_single_rl_sbx_mjx/"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── TensorBoard setup ────────────────────────────────────────────────────
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(tb_dir, f"run_{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(tb_dir, f"run_{run_idx}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(run_dir)
        print(f"[TB] Logging to: {run_dir}")
    except ImportError:
        print("[TB] torch not found — TensorBoard disabled.")
        tb_writer = None

    # ── Curriculum loop ──────────────────────────────────────────────────────
    global_step_counter = 0
    model_path_prev     = None

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        level = stage["level"]
        steps = stage["steps"]
        lr    = stage["lr"]

        print(f"\n{'#' * 60}")
        print(f"  CURRICULUM STAGE {level}  |  {steps:,} steps  |  LR = {lr}")
        print(f"{'#' * 60}\n")

        # Build MJX environment for this curriculum level.
        # A new instance is needed per stage because curriculum_level affects
        # JIT-compiled behaviour (Python-level conditionals in reset/step).
        mjx_env     = KRTIAviaryJax(curriculum_level=level)
        env_cluster = MJXVecEnv(mjx_env, num_envs=NUM_ENVS, seed=42 + stage_idx)

        stage_tb_log      = os.path.join(output_dir, "tensorboard")
        stage_ckpt_prefix = f"sbx_mjx_stage{level}_brain"

        # ── Load or create the SBX PPO model ─────────────────────────────────
        if model_path_prev and os.path.exists(model_path_prev + ".zip"):
            print(
                f"[TRANSFER] Loading stage-{level - 1} model from "
                f"{model_path_prev}.zip ...\n"
            )
            model = PPO.load(
                model_path_prev,
                env=env_cluster,
                tensorboard_log=stage_tb_log,
                custom_objects={
                    "learning_rate": lr,
                    "n_steps":       512,
                    "batch_size":    4096,
                },
            )
            model.learning_rate = lr
        else:
            print("[START FRESH] No prior model — initialising from scratch.\n")
            model = PPO(
                "MlpPolicy",
                env_cluster,
                learning_rate = lr,
                # Rollout buffer: 1024 envs x 512 steps = 524,288 transitions.
                # Reduce n_steps if SBX runs out of GPU/host memory.
                n_steps    = 512,
                batch_size = 4096,   # minibatch size for each PPO gradient step
                n_epochs   = 10,
                gamma      = 0.99,
                verbose    = 1,
                tensorboard_log = stage_tb_log,
                policy_kwargs = dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                ),
            )

        # ── Callbacks ────────────────────────────────────────────────────────
        checkpoint_cb = CheckpointCallback(
            save_freq   = max(50_000 // NUM_ENVS, 1),
            save_path   = checkpoint_dir,
            name_prefix = stage_ckpt_prefix,
        )
        tb_cb = TBMetricsCallback(
            writer        = tb_writer,
            global_offset = global_step_counter,
        )

        # ── Train ─────────────────────────────────────────────────────────────
        model.learn(
            total_timesteps     = steps,
            callback            = [checkpoint_cb, tb_cb],
            tb_log_name         = f"PPO_MJX_stage{level}",
            reset_num_timesteps = (stage_idx == 0),
            progress_bar        = True,
        )

        # ── Save stage final weights ──────────────────────────────────────────
        stage_final = os.path.join(output_dir, f"stage_{level}_final_sbx_mjx_brain")
        model.save(stage_final)
        model_path_prev = stage_final

        print(f"\n{'=' * 60}")
        print(f"  STAGE {level} COMPLETE  |  Saved: {stage_final}.zip")
        print(f"{'=' * 60}")

        global_step_counter += steps
        env_cluster.close()

        # ── Continuation prompt ───────────────────────────────────────────────
        if stage_idx < len(CURRICULUM_STAGES) - 1:
            while True:
                choice = input(
                    f"\nAdvance to curriculum stage {level + 1}? "
                    "[y]es / [n]o / [q]uit: "
                ).strip().lower()
                if choice in ("y", "yes"):
                    print(f"\nInitialising stage {level + 1} ...\n")
                    break
                elif choice in ("n", "no", "q", "quit"):
                    print(
                        f"\nExiting. Progress up to stage {level} safely saved at:\n"
                        f"  {stage_final}.zip\n"
                    )
                    if tb_writer:
                        tb_writer.close()
                    sys.exit(0)
                else:
                    print("Invalid input. Please type 'y' to continue or 'n' to stop.")

    # ── Final combined save ───────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final_krti_sbx_mjx_brain")
    model.save(final_path)
    print(f"\n[DONE] Full curriculum complete. Final model: {final_path}.zip\n")

    if tb_writer:
        tb_writer.close()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
