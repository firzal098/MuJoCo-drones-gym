"""JAX-based Flight Controller for MuJoCo MJX.
Combines ArduPilot-like GUIDED mode integration with PID Control in a stateless JAX format.
"""

import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class ControllerState:
    integral_pos_e: jnp.ndarray
    integral_rpy_e: jnp.ndarray
    last_pos_e: jnp.ndarray
    last_rpy_e: jnp.ndarray
    target_pos: jnp.ndarray
    target_yaw: jnp.ndarray

def init_controller_state(init_pos, init_yaw) -> ControllerState:
    return ControllerState(
        integral_pos_e=jnp.zeros(3),
        integral_rpy_e=jnp.zeros(3),
        last_pos_e=jnp.zeros(3),
        last_rpy_e=jnp.zeros(3),
        target_pos=init_pos,
        target_yaw=init_yaw
    )

def quat_to_rpy(quat):
    """Convert quaternion [w,x,y,z] to RPY."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = jnp.arcsin(jnp.clip(sinp, -1, 1))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return jnp.array([roll, pitch, yaw])

def update_controller(
    dt: float,
    ctrl_state: ControllerState,
    cur_pos: jnp.ndarray,
    cur_quat: jnp.ndarray,
    cur_vel: jnp.ndarray,
    cur_ang_vel: jnp.ndarray,
    target_vel_body: jnp.ndarray,
    target_yaw_rate: jnp.ndarray,
    mass: float,
    gravity: float,
    kf: float,
    km: float,
    arm_length: float,
    max_rpm: float,
):
    """
    Computes RPMs given a body-frame velocity target.
    Stateless pure function designed for jax.jit.
    
    target_vel_body: [vx_body, vy_body, vz_body]
    target_yaw_rate: yaw rate
    """
    cur_rpy = quat_to_rpy(cur_quat)
    yaw = cur_rpy[2]
    
    # 1. Integrate target position based on command velocities
    # Transform body velocity to world velocity
    cos_y, sin_y = jnp.cos(yaw), jnp.sin(yaw)
    vx_world = target_vel_body[0] * cos_y - target_vel_body[1] * sin_y
    vy_world = target_vel_body[0] * sin_y + target_vel_body[1] * cos_y
    vz_world = target_vel_body[2]
    
    target_vel_world = jnp.array([vx_world, vy_world, vz_world])
    
    # Update position and yaw targets (Integrate)
    new_target_pos = ctrl_state.target_pos + target_vel_world * dt
    new_target_yaw = ctrl_state.target_yaw + target_yaw_rate * dt
    new_target_yaw = (new_target_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # Select controller gains dynamically depending on vehicle mass
    p_coeff_for, i_coeff_for, d_coeff_for, p_coeff_tor, i_coeff_tor, d_coeff_tor = jax.lax.cond(
        mass > 1.0,
        lambda _: (
            jnp.array([1.5, 1.5, 2.5]),     # P_COEFF_FOR
            jnp.array([0.05, 0.05, 0.05]),  # I_COEFF_FOR
            jnp.array([1.2, 1.2, 2.5]),     # D_COEFF_FOR
            jnp.array([12.0, 12.0, 6.0]),   # P_COEFF_TOR
            jnp.array([0.0, 0.0, 0.5]),     # I_COEFF_TOR
            jnp.array([3.0, 3.0, 1.5])      # D_COEFF_TOR
        ),
        lambda _: (
            jnp.array([0.4, 0.4, 1.0]),     # P_COEFF_FOR
            jnp.array([0.01, 0.01, 0.01]),  # I_COEFF_FOR
            jnp.array([0.9, 0.9, 2.0]),     # D_COEFF_FOR
            jnp.array([0.002, 0.002, 0.001]),  # P_COEFF_TOR
            jnp.array([0.0, 0.0, 0.0001]),  # I_COEFF_TOR
            jnp.array([0.0005, 0.0005, 0.0002]) # D_COEFF_TOR
        ),
        operand=None
    )
    
    # Position error
    pos_e = new_target_pos - cur_pos
    vel_e = target_vel_world - cur_vel
    
    new_integral_pos_e = ctrl_state.integral_pos_e + pos_e * dt
    new_integral_pos_e = jnp.clip(new_integral_pos_e, -2.0, 2.0)
    
    # Target acceleration
    target_acc = (
        p_coeff_for * pos_e +
        i_coeff_for * new_integral_pos_e +
        d_coeff_for * vel_e
    )
    target_acc = jnp.clip(target_acc, -8.0, 8.0)  # cap commanded accel, tune as needed

    # Target thrust
    # Compensate gravity
    target_acc_z = target_acc[2] + gravity
    # Prevent thrust direction from flipping
    target_acc_z = jnp.maximum(target_acc_z, 0.0)
    target_acc = target_acc.at[2].set(target_acc_z)
    
    thrust = mass * jnp.linalg.norm(target_acc)
    
    # Target attitude
    acc_norm = jnp.linalg.norm(target_acc)
    z_axis = jnp.where(acc_norm > 1e-6, target_acc / acc_norm, jnp.array([0.0, 0.0, 1.0]))
    
    x_c = jnp.array([jnp.cos(new_target_yaw), jnp.sin(new_target_yaw), 0.0])
    y_axis = jnp.cross(z_axis, x_c)
    y_norm = jnp.linalg.norm(y_axis)
    y_axis = jnp.where(y_norm > 1e-6, y_axis / y_norm, jnp.array([0.0, 1.0, 0.0]))
    x_axis = jnp.cross(y_axis, z_axis)
    
    target_roll = jnp.arcsin(jnp.clip(y_axis[2], -1.0, 1.0))
    target_pitch = jnp.arctan2(-x_axis[2], z_axis[2])
    
    rpy_e = jnp.array([target_roll - cur_rpy[0], target_pitch - cur_rpy[1], new_target_yaw - cur_rpy[2]])
    rpy_e = rpy_e.at[2].set((rpy_e[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    
    new_integral_rpy_e = ctrl_state.integral_rpy_e + rpy_e * dt
    new_integral_rpy_e = jnp.clip(new_integral_rpy_e, -0.5, 0.5)
    
    # d_rpy_e = jnp.where(dt > 0, (rpy_e - ctrl_state.last_rpy_e) / dt, jnp.zeros(3))
    d_rpy_e = -cur_ang_vel  # rate damping: oppose actual angular velocity directly

    target_torques = (
        p_coeff_tor * rpy_e +
        i_coeff_tor * new_integral_rpy_e +
        d_coeff_tor * d_rpy_e
    )
    
    # 3. Allocation (X-configuration)
    s2 = jnp.sqrt(2.0)
    km_kf = km / kf
    
    # Scale torques to achievable range
    max_torque_xy = arm_length / s2 * kf * max_rpm**2 * 0.3
    max_torque_z = km_kf * kf * max_rpm**2 * 0.3
    
    tx = jnp.clip(target_torques[0], -max_torque_xy, max_torque_xy)
    ty = jnp.clip(target_torques[1], -max_torque_xy, max_torque_xy)
    tz = jnp.clip(target_torques[2], -max_torque_z, max_torque_z)
    
    # Motor mixing matrix inverse precomputed for efficiency
    l_s2 = arm_length / s2
    
    A = jnp.array([
        [1.0, 1.0, 1.0, 1.0],
        [l_s2, l_s2, -l_s2, -l_s2],
        [-l_s2, l_s2, l_s2, -l_s2],
        [-km_kf, km_kf, -km_kf, km_kf]
    ])
    
    b = jnp.array([thrust, tx, ty, tz])
    motor_forces = jnp.linalg.solve(A, b)
    
    motor_forces = jnp.clip(motor_forces, 0.0, kf * max_rpm**2)
    rpm = jnp.sqrt(motor_forces / kf)
    rpm = jnp.clip(rpm, 0.0, max_rpm)
    
    new_ctrl_state = ControllerState(
        integral_pos_e=new_integral_pos_e,
        integral_rpy_e=new_integral_rpy_e,
        last_pos_e=pos_e,
        last_rpy_e=rpy_e,
        target_pos=new_target_pos,
        target_yaw=new_target_yaw
    )
    
    return rpm, new_ctrl_state
