"""KRTI Arena JAX implementation with simplified goal-directed and path-tracking rewards."""

import jax
import mujoco
import numpy as np
from brax.envs.base import Env, State
from jax import numpy as jnp
from mujoco import mjx

from multi_drone_mujoco.jax_control.guided_mode_jax import (
    init_controller_state,
    update_controller,
)
from multi_drone_mujoco.jax_envs.base_aviary_jax import BaseAviaryJax
from multi_drone_mujoco.utils.enums import DroneModel


def get_gate_corners_jax(gate_type="single"):
    x_min, x_max = 0.07, 1.83
    z_min, z_max = 0.20, 1.85
    t = 0.08
    y_extents = [-t, t]

    corners = []
    for dx in [x_min, x_max]:
        for dy in y_extents:
            for dz in [z_min, z_max]:
                corners.append(jnp.array([dx, dy, dz]))
    return jnp.stack(corners)


def compute_fake_yolo_jax(gate_pos, gate_mat, cam_pos, cam_mat, fovy, width, height):
    """JAX implementation of 3D to 2D camera projection for YOLO bounding box."""
    corners = get_gate_corners_jax()

    def project_pt(pt_local):
        pt_world = jnp.dot(gate_mat, pt_local) + gate_pos
        dp = pt_world - cam_pos
        p_cam = jnp.dot(cam_mat.T, dp)
        x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]

        depth = -z_c
        f_y = 1.0 / jnp.tan(jnp.deg2rad(fovy) / 2.0)
        f_x = f_y * (height / width)

        ndc_x = f_x * (x_c / depth)
        ndc_y = f_y * (y_c / depth)

        px_x = (ndc_x + 1.0) / 2.0 * width
        px_y = (1.0 - ndc_y) / 2.0 * height

        valid = (
            (z_c < 0) & (px_x >= 0) & (px_x <= width) & (px_y >= 0) & (px_y <= height)
        )
        return jnp.array([px_x, px_y]), valid

    pts, valids = jax.vmap(project_pt)(corners)
    any_valid = jnp.any(valids)

    xs = jnp.where(valids, pts[:, 0], jnp.inf)
    ys = jnp.where(valids, pts[:, 1], jnp.inf)

    x_min = jnp.clip(jnp.min(xs) / width, 0.0, 1.0)
    y_min = jnp.clip(jnp.min(ys) / height, 0.0, 1.0)

    xs_max = jnp.where(valids, pts[:, 0], -jnp.inf)
    ys_max = jnp.where(valids, pts[:, 1], -jnp.inf)

    x_max = jnp.clip(jnp.max(xs_max) / width, 0.0, 1.0)
    y_max = jnp.clip(jnp.max(ys_max) / height, 0.0, 1.0)

    valid_box = jnp.array([x_min, y_min, x_max, y_max])
    invalid_box = jnp.array([-1.0, -1.0, -1.0, -1.0])

    return jnp.where(any_valid, valid_box, invalid_box)


class KRTIAviaryJax(BaseAviaryJax):
    def __init__(self, **kwargs):
        from multi_drone_mujoco.envs.base_aviary import DRONE_PARAMS

        model_type = kwargs.get("drone_model", DroneModel.CF2X)
        DRONE_PARAMS[model_type]["mass"] = 3.500
        DRONE_PARAMS[model_type]["ixx"] = 0.080
        DRONE_PARAMS[model_type]["iyy"] = 0.080
        DRONE_PARAMS[model_type]["izz"] = 0.150
        DRONE_PARAMS[model_type]["arm_length"] = np.sqrt(0.130**2 + 0.200**2)
        DRONE_PARAMS[model_type]["kf"] = 0.002
        DRONE_PARAMS[model_type]["km"] = 8.0e-4
        DRONE_PARAMS[model_type]["collision_r"] = 0.235
        DRONE_PARAMS[model_type]["collision_h"] = 0.110
        DRONE_PARAMS[model_type]["prop_radius"] = 0.127

        super().__init__(drone_model=model_type, **kwargs)

        from multi_drone_mujoco.envs.base_aviary import _generate_aviary_xml
        from multi_drone_mujoco.examples.krti_arena import generate_krti_arena_xml

        init_xyzs = np.array([[0.0, 0.0, 0.25]] * self.num_drones)
        init_rpys = np.zeros((self.num_drones, 3))
        xml_str = _generate_aviary_xml(
            self.num_drones,
            self.drone_model,
            init_xyzs,
            init_rpys,
            obstacles=False,
            vision=True,
        )

        xml_str = xml_str.replace('size="10 10 0.05"', 'size="80 80 0.05"')
        xml_str = xml_str.replace('pos="0.17 9.26 0.0"', 'pos="0.17 21.26 0.0"')

        krti_xml = generate_krti_arena_xml()
        krti_xml = krti_xml.replace('contype="1"', 'contype="0"').replace(
            'conaffinity="1"', 'conaffinity="0"'
        )
        xml_str = xml_str.replace("</worldbody>", krti_xml + "\n  </worldbody>")

        self.mj_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mj_model.opt.timestep = 1.0 / self.sim_freq

        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.mj_model.opt.iterations = 8
        self.mj_model.opt.ls_iterations = 8
        self.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        self.cam_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "drone0_cam"
        )
        self.gate_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gate_single_a"
        )
        self.drone_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "drone0"
        )

        self.mass = 3.500
        self.arm_length = jnp.sqrt(0.130**2 + 0.200**2)
        self.kf = 0.002
        self.km = 8.0e-4
        self.max_rpm = 100.0

        self.mj_model.body_mass[self.drone_body_id] = self.mass
        self.sys = mjx.put_model(self.mj_model)

        self.nominal_gate_pos = jnp.array([1.12156, 21.26, 1.0])
        self.nominal_drone_pos = jnp.array([0.92, 24.47, 1.0])

    @property
    def backend(self):
        return "mjx"

    @property
    def observation_size(self):
        return 13

    @property
    def action_size(self):
        return 4

    def reset(self, rng):
        rng, rng_g, rng_d, rng_m = jax.random.split(rng, 4)
        rng_g_dist, rng_g_pos = jax.random.split(rng_g, 2)

        base_gate_y = 24.47 - jax.random.uniform(rng_g_dist, minval=3.0, maxval=8.0)
        nominal_gate_pos = jnp.array([1.12156, base_gate_y, 1.0])

        g_offset = jax.random.uniform(rng_g_pos, (2,), minval=-0.8, maxval=0.8)
        gate_pos = nominal_gate_pos.at[:2].add(g_offset)

        d_offset = jax.random.uniform(rng_d, (2,), minval=-0.6, maxval=0.6)
        drone_pos = self.nominal_drone_pos.at[:2].add(d_offset)
        dyaw = jax.random.uniform(rng_d, minval=-0.3, maxval=0.3)
        yaw = -jnp.pi / 2 + dyaw

        cr = jnp.cos(0.0)
        sr = jnp.sin(0.0)
        cp = jnp.cos(0.0)
        sp = jnp.sin(0.0)
        cy = jnp.cos(yaw / 2)
        sy = jnp.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * sp * sy
        qz = cr * cp * sy - sr * sp * cy
        drone_quat = jnp.array([qw, qx, qy, qz])

        map_noise = jax.random.normal(rng_m, (3,)) * 0.20
        map_noise = map_noise.at[2].set(0.0)
        noisy_gate_pos = gate_pos + map_noise

        qpos = self.sys.qpos0
        qpos = qpos.at[0:3].set(drone_pos)
        qpos = qpos.at[3:7].set(drone_quat)

        hover_rpm = jnp.sqrt((self.mass * 9.81) / (4 * self.kf))
        qvel = jnp.zeros(self.sys.nv)
        d = mjx.make_data(self.sys)
        d = d.replace(qpos=qpos, qvel=qvel)
        d = mjx.step(self.sys, d)

        ctrl_state = init_controller_state(drone_pos, yaw)

        info = {
            "gate_pos": gate_pos,
            "noisy_gate_pos": noisy_gate_pos,
            "ctrl_state": ctrl_state,
            "step": 0,
            "rng": rng,
        }

        obs = self._get_obs(d, info)

        metrics = {
            "crashed": jnp.zeros(()),
            "cleared_gate": jnp.zeros(()),
            "gate_collided": jnp.zeros(()),
        }

        return State(
            d, obs, jnp.zeros(()), jnp.zeros((), dtype=jnp.float32), metrics, info
        )

    def _get_obs(self, d, info):
        drone_pos = d.qpos[0:3]
        drone_quat = d.qpos[3:7]
        vx_world, vy_world, vz_world = d.qvel[0:3]

        w, x, y, z = drone_quat
        roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        rel_gate_world = info["noisy_gate_pos"] - drone_pos
        cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)

        rel_gate_body = jnp.array(
            [
                rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,
                -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y),
                -rel_gate_world[2],
            ]
        )

        vel_body = jnp.array(
            [
                vx_world * cos_y + vy_world * sin_y,
                -(-vx_world * sin_y + vy_world * cos_y),
                -vz_world,
            ]
        )

        cam_pos = d.cam_xpos[self.cam_id]
        cam_mat = d.cam_xmat[self.cam_id].reshape(3, 3)
        gate_mat = jnp.eye(3)

        rng, subrng = jax.random.split(info["rng"])
        drop = jax.random.uniform(subrng) < 0.05

        gate_body_pos = info["gate_pos"].at[0].add(-0.95156).at[2].set(0.0)
        yolo_raw = compute_fake_yolo_jax(
            gate_body_pos, gate_mat, cam_pos, cam_mat, 60.0, 320, 240
        )
        yolo_raw = jnp.where(drop, jnp.array([-1.0, -1.0, -1.0, -1.0]), yolo_raw)

        def valid_yolo(y):
            err_x = (y[0] + y[2]) / 2.0 - 0.5
            err_y = (y[1] + y[3]) / 2.0 - 0.5
            return jnp.array([err_x, err_y, y[2] - y[0], y[3] - y[1]])

        yolo_obs = jax.lax.cond(
            yolo_raw[0] >= 0.0,
            lambda y: valid_yolo(y),
            lambda y: jnp.array([0.0, 0.0, 0.0, 0.0]),
            yolo_raw,
        )

        yaw_error = jnp.arctan2(rel_gate_body[1], rel_gate_body[0])

        rel_gate_body_scaled = jnp.clip(rel_gate_body / 20.0, -1.0, 1.0)
        vel_body_scaled = jnp.clip(vel_body / 4.0, -1.0, 1.0)
        attitude_scaled = jnp.clip(
            jnp.array([roll, -pitch, yaw_error]) / jnp.pi, -1.0, 1.0
        )

        return jnp.concatenate(
            [yolo_obs, rel_gate_body_scaled, vel_body_scaled, attitude_scaled]
        )

    def step(self, state, action):
        max_xy_speed = 4.0
        max_z_speed = 2.5
        max_yaw_rate = 2.0
        MAX_STEPS = 400
        vx_body = action[0] * max_xy_speed
        vy_body = -action[1] * max_xy_speed
        vz_body = -action[2] * max_z_speed
        yaw_rate = -action[3] * max_yaw_rate

        target_vel = jnp.array([vx_body, vy_body, vz_body])
        d = state.pipeline_state

        rpm, next_ctrl_state = update_controller(
            self.dt,
            state.info["ctrl_state"],
            d.qpos[0:3],
            d.qpos[3:7],
            d.qvel[0:3],
            d.qvel[3:6],
            target_vel,
            yaw_rate,
            self.mass,
            self.G,
            self.kf,
            self.km,
            self.arm_length,
            self.max_rpm,
        )

        forces = (rpm**2) * self.kf
        torques = (rpm**2) * self.km
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        total_thrust = jnp.sum(forces)

        L = self.arm_length / jnp.sqrt(2.0)
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * L
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * L

        def body_fn(i, cur_d):
            rot_mat = cur_d.xmat[self.drone_body_id].reshape(3, 3)
            thrust_world = jnp.dot(rot_mat, jnp.array([0.0, 0.0, total_thrust]))
            torque_world = jnp.dot(rot_mat, jnp.array([x_torque, y_torque, z_torque]))

            force_torque = jnp.concatenate([thrust_world, torque_world])
            cur_d = cur_d.replace(
                xfrc_applied=cur_d.xfrc_applied.at[self.drone_body_id].set(force_torque)
            )

            cur_d = mjx.step(self.sys, cur_d)
            return cur_d

        d = jax.lax.fori_loop(0, self.sim_steps_per_ctrl, body_fn, d)

        dist_before = jnp.linalg.norm(
            state.info["gate_pos"] - state.pipeline_state.qpos[0:3]
        )
        dist_after = jnp.linalg.norm(state.info["gate_pos"] - d.qpos[0:3])

        # Calculate body-frame velocity for hover penalty
        vx_world, vy_world, vz_world = d.qvel[0:3]
        w_q, x_q, y_q, z_q = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        yaw = jnp.arctan2(2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q))
        cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)
        vel_body = jnp.array(
            [
                vx_world * cos_y + vy_world * sin_y,
                -(-vx_world * sin_y + vy_world * cos_y),
                -vz_world,
            ]
        )

        # 1. Progression Reward
        raw_progression = (dist_before - dist_after) * 100.0
        progression_reward = jnp.where(
            raw_progression < 0.0, raw_progression * 0.1, raw_progression
        )

        # 2. Velocity Alignment Reward - encourage flying fast toward gate
        rel_vector = state.info["gate_pos"] - d.qpos[0:3]
        unit_direction_to_gate = rel_vector / (jnp.linalg.norm(rel_vector) + 1e-6)
        actual_velocity = d.qvel[0:3]
        velocity_alignment = jnp.dot(actual_velocity, unit_direction_to_gate)
        alignment_reward = velocity_alignment * 2.0

        # 2b. Yaw-facing-gate reward: reward for pointing at gate center
        w_q, x_q, y_q, z_q = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        drone_yaw = jnp.arctan2(
            2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q)
        )
        desired_yaw = jnp.arctan2(rel_vector[1], rel_vector[0])
        yaw_diff = jnp.abs((drone_yaw - desired_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        yaw_reward = jnp.cos(yaw_diff) * 3.0

        # 3. Path Centerline Tracking Reward - moderate proximity penalty
        lateral_error = d.qpos[0] - state.info["gate_pos"][0]
        height_error = d.qpos[2] - state.info["gate_pos"][2]

        dist_y = jnp.maximum(d.qpos[1] - state.info["gate_pos"][1], 0.0)
        tracking_weight = 3.0 + 12.0 / (dist_y + 0.5)
        tracking_reward = -tracking_weight * (
            jnp.square(lateral_error) + jnp.square(height_error)
        )

        # 4. Hover Penalty (discourage staying still hovering)
        hover_penalty = -jnp.exp(-jnp.linalg.norm(vel_body) * 3.0) * 0.5

        crossed_plane = (state.pipeline_state.qpos[1] > state.info["gate_pos"][1]) & (
            d.qpos[1] <= state.info["gate_pos"][1]
        )
        outside_opening = (
            (d.qpos[0] < state.info["gate_pos"][0] - 0.85)
            | (d.qpos[0] > state.info["gate_pos"][0] + 0.85)
            | (d.qpos[2] < 0.20)
            | (d.qpos[2] > 1.85)
        )

        is_gate_crash = crossed_plane & outside_opening
        cleared_gate = crossed_plane & (~outside_opening)

        w, x, y, z = d.qpos[3:7]
        roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))

        is_generic_crash = (
            (d.qpos[2] < 0.08)
            | (d.qpos[2] > 4.5)
            | (jnp.abs(roll) > jnp.pi / 2.2)
            | (jnp.abs(pitch) > jnp.pi / 2.2)
        )

        is_timeout_step = (state.info["step"] + 1) >= MAX_STEPS

        accuracy_bonus = (0.50 - dist_after) * 1000.0
        accuracy_bonus = jnp.clip(accuracy_bonus, 0.0, 500.0)

        terminal_reward = jnp.where(cleared_gate, 1000.0 + accuracy_bonus, 0.0)
        terminal_reward += jnp.where(is_gate_crash, -800.0, 0.0)

        generic_crash_penalty = -500.0 - jnp.clip(dist_after * 20.0, 0.0, 200.0)
        terminal_reward += jnp.where(
            is_generic_crash & (~is_gate_crash), generic_crash_penalty, 0.0
        )

        ran_out_the_clock = (
            is_timeout_step & (~cleared_gate) & (~is_generic_crash) & (~is_gate_crash)
        )
        terminal_reward += jnp.where(ran_out_the_clock, -400.0, 0.0)

        time_penalty = -1.0
        reward = (
            progression_reward
            + alignment_reward
            + yaw_reward
            + tracking_reward
            + hover_penalty
            + time_penalty
            + terminal_reward
        )

        done = cleared_gate | is_gate_crash | is_generic_crash

        new_info = dict(state.info)
        rng, subrng = jax.random.split(state.info["rng"])
        new_info["ctrl_state"] = next_ctrl_state
        new_info["step"] = state.info["step"] + 1
        new_info["rng"] = rng

        obs = self._get_obs(d, new_info)
        done = (done | (new_info["step"] >= MAX_STEPS)).astype(jnp.float32)

        metrics = dict(state.metrics)
        metrics.update(
            {
                "crashed": (is_generic_crash | is_gate_crash).astype(jnp.float32),
                "cleared_gate": cleared_gate.astype(jnp.float32),
                "gate_collided": is_gate_crash.astype(jnp.float32),
            }
        )

        return state.replace(
            pipeline_state=d,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )
