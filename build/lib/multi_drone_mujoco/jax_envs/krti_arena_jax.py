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

        # Ensure the point is in front of the camera lens (depth > 0).
        valid = z_c < 0
        return jnp.array([px_x, px_y]), valid

    pts, valids = jax.vmap(project_pt)(corners)
    any_valid = jnp.any(valids)

    # Use infinity masking for invalid points during min/max reductions
    xs = jnp.where(valids, pts[:, 0], jnp.inf)
    ys = jnp.where(valids, pts[:, 1], jnp.inf)

    unclipped_xmin = jnp.min(xs)
    unclipped_ymin = jnp.min(ys)

    xs_max = jnp.where(valids, pts[:, 0], -jnp.inf)
    ys_max = jnp.where(valids, pts[:, 1], -jnp.inf)

    unclipped_xmax = jnp.max(xs_max)
    unclipped_ymax = jnp.max(ys_max)

    # Verify if the unclipped bounding box has any physical overlap with the screen viewport.
    outside = (unclipped_xmin > width) | (unclipped_xmax < 0.0) | (unclipped_ymin > height) | (unclipped_ymax < 0.0)
    visible = any_valid & (~outside)

    # Scale minimum/maximum coordinates to normalized space [0.0, 1.0]
    x_min = jnp.clip(unclipped_xmin / width, 0.0, 1.0)
    y_min = jnp.clip(unclipped_ymin / height, 0.0, 1.0)
    x_max = jnp.clip(unclipped_xmax / width, 0.0, 1.0)
    y_max = jnp.clip(unclipped_ymax / height, 0.0, 1.0)

    valid_box = jnp.array([x_min, y_min, x_max, y_max])
    invalid_box = jnp.array([-1.0, -1.0, -1.0, -1.0])

    return jnp.where(visible, valid_box, invalid_box)


class KRTIAviaryJax(BaseAviaryJax):
    def __init__(self, curriculum_level=1, **kwargs):
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

        # Redesigned curriculum configuration parameter check
        self.curriculum_level = curriculum_level

        # Phase 1 and 2 maximum limits scaling dynamically
        if self.curriculum_level in [1, 2]:
            self.max_xy_speed = 4.0
            self.max_z_speed = 2.0
            self.max_yaw_rate = 1.0
        elif self.curriculum_level == 3:
            self.max_xy_speed = 6.0
            self.max_z_speed = 2.5
            self.max_yaw_rate = 1.5
        else:
            self.max_xy_speed = 10.0
            self.max_z_speed = 3.0
            self.max_yaw_rate = 2.0

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
        return 20

    @property
    def action_size(self):
        return 4


    def _get_yolo_and_direction(self, d, gate_pos, noisy_gate_pos, rng_key, last_yolo_mem, init_flag=False):
        """Purity-preserving visual tracking projection with JAX-native memory and fallback recovery steering."""
        cam_pos = d.cam_xpos[self.cam_id]
        cam_mat = d.cam_xmat[self.cam_id].reshape(3, 3)
        gate_mat = jnp.eye(3)

        gate_body_pos = gate_pos.at[0].add(-0.95156).at[2].set(0.0)
        yolo_raw = compute_fake_yolo_jax(
            gate_body_pos, gate_mat, cam_pos, cam_mat, 60.0, 320, 240
        )
        
        # Camera dropout simulation activated only on stage 4 and above
        enable_dropout = self.curriculum_level >= 4
        drop = jax.lax.cond(
            enable_dropout & (~init_flag),
            lambda: jax.random.uniform(rng_key) < 0.05,
            lambda: False
        )
        yolo_raw = jnp.where(drop, jnp.array([-1.0, -1.0, -1.0, -1.0]), yolo_raw)
        
        yolo_visible = yolo_raw[0] >= 0.0
        
        # Build coordinates
        curr_u_c = (yolo_raw[0] + yolo_raw[2]) - 1.0
        curr_v_c = (yolo_raw[1] + yolo_raw[3]) - 1.0
        curr_w_b = yolo_raw[2] - yolo_raw[0]
        curr_h_b = yolo_raw[3] - yolo_raw[1]
        
        # Save or fetch historical visual inputs (YOLO Memory)
        new_last_yolo = jnp.where(
            yolo_visible,
            jnp.array([curr_u_c, curr_v_c, curr_w_b, curr_h_b]),
            last_yolo_mem
        )
        
        u_c_obs = jnp.where(yolo_visible, curr_u_c, new_last_yolo[0])
        v_c_obs = jnp.where(yolo_visible, curr_v_c, new_last_yolo[1])
        w_b_obs = jnp.where(yolo_visible, curr_w_b, new_last_yolo[2])
        h_b_obs = jnp.where(yolo_visible, curr_h_b, new_last_yolo[3])
        f_vis = jnp.where(yolo_visible, 1.0, 0.0)
        
        yolo_obs = jnp.array([u_c_obs, v_c_obs, w_b_obs, h_b_obs, f_vis])
        
        # Compute body relative direction targeting
        drone_pos = d.qpos[0:3]
        drone_quat = d.qpos[3:7]
        w, x, y, z = drone_quat
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)
        
        rel_gate_world = noisy_gate_pos - drone_pos
        rel_gate_body = jnp.array([
            rel_gate_world[0] * cos_y + rel_gate_world[1] * sin_y,
            -(-rel_gate_world[0] * sin_y + rel_gate_world[1] * cos_y),
            -rel_gate_world[2],
        ])
        
        dir_gate_body = rel_gate_body / (jnp.linalg.norm(rel_gate_body) + 1e-6)
        
        return yolo_obs, new_last_yolo, dir_gate_body


    def reset(self, rng):
        rng, rng_g, rng_d, rng_m = jax.random.split(rng, 4)
        rng_g_dist, rng_g_pos = jax.random.split(rng_g, 2)

        # Apply specific randomization policies based on Stage level (Point 9)
        if self.curriculum_level == 1:
            # Stage 1: Fixed Drone, Fixed Gate, Fixed Orientation
            gate_pos = self.nominal_gate_pos
            drone_pos = self.nominal_drone_pos
            yaw = -jnp.pi / 2
        elif self.curriculum_level == 2:
            # Stage 2: Tight Randomization (±0.2m position, ±5° Yaw)
            g_offset = jax.random.uniform(rng_g_pos, (2,), minval=-0.2, maxval=0.2)
            gate_pos = self.nominal_gate_pos.at[:2].add(g_offset)
            d_offset = jax.random.uniform(rng_d, (2,), minval=-0.2, maxval=0.2)
            drone_pos = self.nominal_drone_pos.at[:2].add(d_offset).at[2].set(1.0)
            dyaw = jax.random.uniform(rng_d, minval=-0.087, maxval=0.087)
            yaw = -jnp.pi / 2 + dyaw
        elif self.curriculum_level == 3:
            # Stage 3: Moderate Randomization (±0.5m position, ±10° Yaw)
            g_offset = jax.random.uniform(rng_g_pos, (2,), minval=-0.5, maxval=0.5)
            gate_pos = self.nominal_gate_pos.at[:2].add(g_offset)
            d_offset = jax.random.uniform(rng_d, (2,), minval=-0.5, maxval=0.5)
            drone_pos = self.nominal_drone_pos.at[:2].add(d_offset).at[2].set(1.0)
            dyaw = jax.random.uniform(rng_d, minval=-0.174, maxval=0.174)
            yaw = -jnp.pi / 2 + dyaw
        elif self.curriculum_level == 4:
            # Stage 4: High Randomization (±1.0m position, ±20° Yaw)
            g_offset = jax.random.uniform(rng_g_pos, (2,), minval=-1.0, maxval=1.0)
            gate_pos = self.nominal_gate_pos.at[:2].add(g_offset)
            d_offset = jax.random.uniform(rng_d, (2,), minval=-1.0, maxval=1.0)
            drone_pos = self.nominal_drone_pos.at[:2].add(d_offset).at[2].set(1.0)
            dyaw = jax.random.uniform(rng_d, minval=-0.349, maxval=0.349)
            yaw = -jnp.pi / 2 + dyaw
        else:
            # Stage 5: Full Variable Competition Domain with varying baseline distances
            base_gate_y = 24.47 - jax.random.uniform(rng_g_dist, minval=8.0, maxval=18.0)
            nominal_gate_pos = jnp.array([0.17, base_gate_y, 1.0])
            g_offset = jax.random.uniform(rng_g_pos, (2,), minval=-1.0, maxval=1.0)
            gate_pos = nominal_gate_pos.at[:2].add(g_offset)
            d_offset = jax.random.uniform(rng_d, (2,), minval=-1.0, maxval=1.0)
            drone_pos = self.nominal_drone_pos.at[:2].add(d_offset).at[2].set(1.0)
            dyaw = jax.random.uniform(rng_d, minval=-0.3, maxval=0.3)
            yaw = -jnp.pi / 2 + dyaw

        enable_map_noise = self.curriculum_level >= 5
        map_noise = jax.lax.cond(
            enable_map_noise,
            lambda: jax.random.normal(rng_m, (3,)) * 0.20,
            lambda: jnp.zeros(3)
        )
        map_noise = map_noise.at[2].set(0.0)
        noisy_gate_pos = gate_pos + map_noise

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

        qpos = self.sys.qpos0
        qpos = qpos.at[0:3].set(drone_pos)
        qpos = qpos.at[3:7].set(drone_quat)

        qvel = jnp.zeros(self.sys.nv)
        d = mjx.make_data(self.sys)
        d = d.replace(qpos=qpos, qvel=qvel)
        d = mjx.step(self.sys, d)

        ctrl_state = init_controller_state(drone_pos, yaw)


        def stabilize_step(i, val):
            cur_d, cur_ctrl_state = val
            pos = cur_d.qpos[0:3]
            quat = cur_d.qpos[3:7]
            vel = cur_d.qvel[0:3]
            omega = cur_d.qvel[3:6]
            
            rpm, next_ctrl = update_controller(
                self.dt,
                cur_ctrl_state,
                pos,
                quat,
                vel,
                omega,
                jnp.zeros(3),
                0.0,
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

            def body_fn(j, inner_d):
                rot_mat = inner_d.xmat[self.drone_body_id].reshape(3, 3)
                thrust_world = jnp.dot(rot_mat, jnp.array([0.0, 0.0, total_thrust]))
                torque_world = jnp.dot(rot_mat, jnp.array([x_torque, y_torque, z_torque]))

                force_torque = jnp.concatenate([thrust_world, torque_world])
                inner_d = inner_d.replace(
                    xfrc_applied=inner_d.xfrc_applied.at[self.drone_body_id].set(force_torque)
                )

                inner_d = mjx.step(self.sys, inner_d)
                return inner_d

            next_d = jax.lax.fori_loop(0, self.sim_steps_per_ctrl, body_fn, cur_d)
            return next_d, next_ctrl

        d, ctrl_state = jax.lax.fori_loop(0, 40, stabilize_step, (d, ctrl_state))

        rng, subrng = jax.random.split(rng)
        yolo_obs, init_last_yolo, dir_gate_body = self._get_yolo_and_direction(
            d, gate_pos, noisy_gate_pos, subrng, jnp.zeros(4), init_flag=True
        )

        info = {
            "gate_pos": gate_pos,
            "noisy_gate_pos": noisy_gate_pos,
            "ctrl_state": ctrl_state,
            "step": 0,
            "rng": rng,
            "last_yolo": init_last_yolo,
            "prev_action": jnp.zeros(4),
        }

        # Pack initial proprioceptive variables
        vx_world, vy_world, vz_world = d.qvel[0:3]
        roll = jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = jnp.arcsin(jnp.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
        cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)

        vel_body = jnp.array([
            vx_world * cos_y + vy_world * sin_y,
            -(-vx_world * sin_y + vy_world * cos_y),
            -vz_world,
        ])

        vel_body_scaled = jnp.clip(vel_body / 10.0, -1.0, 1.0)
        omega_body = d.qvel[3:6]
        omega_body_scaled = jnp.clip(omega_body / jnp.pi, -1.0, 1.0)

        roll_scaled = jnp.clip(roll / (jnp.pi / 2.5), -1.0, 1.0)
        pitch_scaled = jnp.clip(pitch / (jnp.pi / 2.5), -1.0, 1.0)

        obs = jnp.concatenate([
            yolo_obs,          # 5
            vel_body_scaled,   # 3
            omega_body_scaled, # 3
            jnp.array([roll_scaled, pitch_scaled]), # 2
            dir_gate_body,      # 3
            info["prev_action"]      # 4
        ])

        metrics = {
            "crashed": jnp.zeros(()),
            "cleared_gate": jnp.zeros(()),
            "gate_collided": jnp.zeros(()),
            "gate_distance": jnp.linalg.norm(gate_pos - drone_pos),
            "reward_progress": jnp.zeros(()),
            "reward_centering": jnp.zeros(()),
            "reward_speed": jnp.zeros(()),
            "reward_attitude": jnp.zeros(()),
            "reward_smooth": jnp.zeros(()),
            "reward_terminal": jnp.zeros(()),
        }

        return State(
            d, obs, jnp.zeros(()), jnp.zeros((), dtype=jnp.float32), metrics, info
        )


    def _get_obs(self, d, info):
        prev_last_yolo = info["last_yolo"]
        
        rng, subrng = jax.random.split(info["rng"])
        yolo_obs, _, dir_gate_body = self._get_yolo_and_direction(
            d, info["gate_pos"], info["noisy_gate_pos"], subrng, prev_last_yolo, init_flag=False
        )

        drone_quat = d.qpos[3:7]
        vx_world, vy_world, vz_world = d.qvel[0:3]

        w, x, y, z = drone_quat
        roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)

        vel_body = jnp.array([
            vx_world * cos_y + vy_world * sin_y,
            -(-vx_world * sin_y + vy_world * cos_y),
            -vz_world,
        ])

        vel_body_scaled = jnp.clip(vel_body / 10.0, -1.0, 1.0)
        omega_body = d.qvel[3:6]
        omega_body_scaled = jnp.clip(omega_body / jnp.pi, -1.0, 1.0)

        roll_scaled = jnp.clip(roll / (jnp.pi / 2.5), -1.0, 1.0)
        pitch_scaled = jnp.clip(pitch / (jnp.pi / 2.5), -1.0, 1.0)

        return jnp.concatenate([
            yolo_obs,          # 5
            vel_body_scaled,   # 3
            omega_body_scaled, # 3
            jnp.array([roll_scaled, pitch_scaled]), # 2
            dir_gate_body,      # 3
            info["prev_action"]      # 4
        ])


    def step(self, state, action):
        # Scale inputs using dynamic limits corresponding to stage curriculum level
        vx_body = action[0] * self.max_xy_speed
        vy_body = -action[1] * self.max_xy_speed
        vz_body = -action[2] * self.max_z_speed
        yaw_rate = -action[3] * self.max_yaw_rate
        MAX_STEPS = 450
        
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

        # Extract pitch, roll, yaw angles
        w_q, x_q, y_q, z_q = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        roll = jnp.arctan2(2 * (w_q * x_q + y_q * z_q), 1 - 2 * (x_q * x_q + y_q * y_q))
        pitch = jnp.arcsin(jnp.clip(2 * (w_q * y_q - z_q * x_q), -1.0, 1.0))
        yaw = jnp.arctan2(2 * (w_q * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q))

        # Update tracking keys
        new_info = dict(state.info)
        rng, subrng = jax.random.split(state.info["rng"])
        new_info["ctrl_state"] = next_ctrl_state
        new_info["step"] = state.info["step"] + 1
        new_info["rng"] = rng

        # Carry forward previous coordinate memories if tracking is lost
        prev_last_yolo = state.info["last_yolo"]
        _, updated_last_yolo, _ = self._get_yolo_and_direction(
            d, state.info["gate_pos"], state.info["noisy_gate_pos"], subrng, prev_last_yolo, init_flag=False
        )
        new_info["last_yolo"] = updated_last_yolo
        new_info["prev_action"] = action

        # Compute updated observation space
        obs = self._get_obs(d, new_info)


        # 1. Potential Distance Reward (Point 2)
        progression_reward = 5.0 * (dist_before - dist_after)

        # 2. Camera Centering Zone Reward (Points 3, 4) - active under 3m
        u_c = obs[0]
        v_c = obs[1]
        f_vis = obs[4]
        r_perception = jnp.where(dist_after < 3.0, 2.0 * f_vis * (1.0 - jnp.square(u_c) - jnp.square(v_c)), 0.0)

        # 3. Dynamic Velocity Limiting (Point 5) - active under 2.0m to slow down near center
        speed = jnp.linalg.norm(d.qvel[0:3])
        speed_err = jnp.square(speed - 2.0)
        reward_speed = jnp.where(dist_after < 2.0, -jnp.exp(jnp.clip(speed_err, 0.0, 3.0)), 0.0)

        # 4. Continuous Safety Attitude Penalties (Point 6)
        reward_attitude = -0.2 * jnp.square(roll) - 0.2 * jnp.square(pitch)

        # 5. Smooth Action Change Penalties (Point 7)
        action_change = action - state.info["prev_action"]
        reward_smooth = -0.02 * jnp.sum(jnp.square(action_change))

        # Static survival credit to avoid suicide loops
        survival_reward = 0.10

        step_reward = progression_reward + r_perception + reward_speed + reward_attitude + reward_smooth + survival_reward


        crossed_plane = (state.pipeline_state.qpos[1] > state.info["gate_pos"][1]) & (
            d.qpos[1] <= state.info["gate_pos"][1]
        )
        
        # Boundary constraints matching real gate passage profile
        within_gate_frame = (
            (d.qpos[0] >= state.info["gate_pos"][0] - 0.88) & 
            (d.qpos[0] <= state.info["gate_pos"][0] + 0.88) & 
            (d.qpos[2] >= state.info["gate_pos"][2] - 0.82) &
            (d.qpos[2] <= state.info["gate_pos"][2] + 0.82)
        )

        in_gate_y_zone = jnp.abs(d.qpos[1] - state.info["gate_pos"][1]) < 0.25

        hits_left_post = (
            (d.qpos[0] < state.info["gate_pos"][0] - 0.70) & 
            (d.qpos[0] > state.info["gate_pos"][0] - 1.10) & 
            (d.qpos[2] < 1.85)
        )

        hits_right_post = (
            (d.qpos[0] > state.info["gate_pos"][0] + 0.70) & 
            (d.qpos[0] < state.info["gate_pos"][0] + 1.10) & 
            (d.qpos[2] < 1.85)
        )

        hits_top_arch = (
            (d.qpos[0] >= state.info["gate_pos"][0] - 1.10) & 
            (d.qpos[0] <= state.info["gate_pos"][0] + 1.10) & 
            (d.qpos[2] > 1.77) &
            (d.qpos[2] < 1.99)
        )

        is_gate_crash = in_gate_y_zone & (hits_left_post | hits_right_post | hits_top_arch)

        is_generic_crash = (
            (d.qpos[2] < 0.08)
            | (d.qpos[2] > 4.5)
            | (jnp.abs(roll) > jnp.pi / 2.5)
            | (jnp.abs(pitch) > jnp.pi / 2.5)
        )

        # Better success validation: plane crossed forward, centering inside frame, and clean flight
        velocity_forward = d.qvel[1] < 0.0  # passage goes along negative Y
        cleared_gate = crossed_plane & within_gate_frame & velocity_forward & (~is_gate_crash) & (~is_generic_crash)

        # Milestone definitions
        terminal_reward = jnp.where(cleared_gate, 100.0, 0.0)
        terminal_reward += jnp.where(is_gate_crash | is_generic_crash, -100.0, 0.0)

        reward = step_reward + terminal_reward

        done = cleared_gate | is_gate_crash | is_generic_crash
        done = (done | (new_info["step"] >= MAX_STEPS)).astype(jnp.float32)

        # Track gate distance metric & subcomponents (Point 10)
        metrics = dict(state.metrics)
        metrics.update(
            {
                "crashed": (is_generic_crash | is_gate_crash).astype(jnp.float32),
                "cleared_gate": cleared_gate.astype(jnp.float32),
                "gate_collided": is_gate_crash.astype(jnp.float32),
                "gate_distance": dist_after,
                "reward_progress": progression_reward,
                "reward_centering": r_perception,
                "reward_speed": reward_speed,
                "reward_attitude": reward_attitude,
                "reward_smooth": reward_smooth,
                "reward_terminal": terminal_reward,
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