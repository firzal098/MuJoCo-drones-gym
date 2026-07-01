import jax
jax.config.update("jax_compilation_cache_dir", "/home/firza/MuJoCo-drones-gym/.jax_cache")
import jax.numpy as jnp
import time
from multi_drone_mujoco.jax_envs.krti_arena_jax import KRTIAviaryJax

env = KRTIAviaryJax()
print("mass:", env.mj_model.body_mass[env.drone_body_id])

rng = jax.random.PRNGKey(42)
state = jax.jit(env.reset)(rng)
jax.block_until_ready(state.pipeline_state.qpos)

t0 = time.time()
action = jnp.zeros(4)
next_state = jax.jit(env.step)(state, action)
jax.block_until_ready(next_state.pipeline_state.qpos)
print(f"step done in {time.time()-t0:.2f}s")