"""ArduPilot GUIDED mode flight controller wrapper for MuJoCo drones."""

import numpy as np
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl

class GuidedVehicle:
    """Wrapper providing ArduPilot-like GUIDED mode control functions."""
    
    def __init__(self, env, drone_index=0, verbose=False):
        self.env = env
        self.drone_idx = drone_index
        self.verbose = verbose
        self.ctrl = DSLPIDControl(env)
        
        # If the drone mass is scaled (e.g. Iris), use appropriate control gains
        if env.M > 1.0:
            self.ctrl.P_COEFF_TOR = np.array([12.0, 12.0, 6.0])
            self.ctrl.I_COEFF_TOR = np.array([0.0, 0.0, 0.5])
            self.ctrl.D_COEFF_TOR = np.array([3.0, 3.0, 1.5])
            self.ctrl.P_COEFF_FOR = np.array([1.5, 1.5, 2.5])
            self.ctrl.I_COEFF_FOR = np.array([0.05, 0.05, 0.05])
            self.ctrl.D_COEFF_FOR = np.array([1.2, 1.2, 2.5])
            self.ctrl.MAX_POS_RATE = 4.0
        
        # Guided State variables
        self._mode = "STABILIZE"  # Default mode
        self._armed = False
        
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.target_vel = np.zeros(3)
        self.target_yaw_rate = 0.0
        
        # Sub-mode control variables
        self._takeoff_altitude = None
        self._descending = False
        self._landing_descend_rate = 0.4  # m/s vertical descent speed
        
    @property
    def mode(self):
        return self._mode
        
    @mode.setter
    def mode(self, new_mode):
        new_mode = new_mode.upper()
        valid_modes = ["STABILIZE", "GUIDED", "TAKEOFF", "LAND"]
        if new_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {new_mode}. Select from {valid_modes}")
        
        if new_mode == self._mode:
            return
        
        # Mode transitions
        if new_mode == "LAND":
            # Target current x, y, and set z to ground level
            self.target_pos[0] = self.env.pos[self.drone_idx, 0]
            self.target_pos[1] = self.env.pos[self.drone_idx, 1]
            self.target_pos[2] = self.env.pos[self.drone_idx, 2] # starts at current alt
            self.target_vel = np.array([0.0, 0.0, -self._landing_descend_rate])
        elif new_mode == "GUIDED":
            # Lock position to current position
            self.target_pos = self.env.pos[self.drone_idx].copy()
            self.target_yaw = self.env.rpy[self.drone_idx, 2]
            self.target_vel = np.zeros(3)
            self.target_yaw_rate = 0.0
            
        self._mode = new_mode
        if self.verbose:
            print(f"[GUIDED] Flight Mode transitioned to: {self._mode}")
        
    @property
    def armed(self):
        return self._armed
        
    @armed.setter
    def armed(self, state: bool):
        self._armed = state
        if not state:
            self.ctrl.reset()
            self.mode = "STABILIZE"
        if self.verbose:
            print(f"[GUIDED] Armed status set to: {self._armed}")
        
    def arm(self):
        self.armed = True
        
    def disarm(self):
        self.armed = False
        
    def simple_takeoff(self, target_altitude):
        """Climb vertically to target_altitude in meters."""
        if not self._armed:
            print("[GUIDED] WARNING: Takeoff command rejected, vehicle is disarmed!")
            return False
            
        self.target_pos[0] = self.env.pos[self.drone_idx, 0]
        self.target_pos[1] = self.env.pos[self.drone_idx, 1]
        self.target_pos[2] = target_altitude
        self.target_yaw = self.env.rpy[self.drone_idx, 2]
        self.target_vel = np.zeros(3)
        self.target_yaw_rate = 0.0
        
        self.mode = "TAKEOFF"
        self._takeoff_altitude = target_altitude
        return True
        
    def simple_goto(self, target_pos, target_yaw=None):
        """Fly to local [X, Y, Z] target position."""
        if not self._armed:
            print("[GUIDED] WARNING: simple_goto ignored, vehicle is disarmed!")
            return False
            
        self.mode = "GUIDED"
        self.target_pos = np.array(target_pos)
        self.target_vel = np.zeros(3)
        
        if target_yaw is not None:
            self.target_yaw = target_yaw
        else:
            # Face target waypoint dynamically if far enough
            dist_xy = np.linalg.norm(self.target_pos[:2] - self.env.pos[self.drone_idx, :2])
            if dist_xy > 0.3:
                self.target_yaw = np.arctan2(
                    self.target_pos[1] - self.env.pos[self.drone_idx, 1],
                    self.target_pos[0] - self.env.pos[self.drone_idx, 0]
                )
        return True
        
    def set_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """Hold velocities (m/s) and yaw rate (rad/s) dynamically."""
        if not self._armed:
            print("[GUIDED] WARNING: set_velocity ignored, vehicle is disarmed!")
            return False
            
        self.mode = "GUIDED"
        self.target_vel = np.array([vx, vy, vz])
        self.target_yaw_rate = yaw_rate
        return True
        
    def land(self):
        """Descend at current position and disarm upon touchdown."""
        if not self._armed:
            print("[GUIDED] WARNING: land ignored, vehicle is already disarmed!")
            return False
        self.mode = "LAND"
        return True
        
    def update(self, control_timestep):
        """Computes motor RPM outputs. Returns ndarray (4,)."""
        if not self._armed:
            return np.zeros(4)
            
        # Fetch current state
        cur_pos = self.env.pos[self.drone_idx]
        cur_quat = self.env.quat[self.drone_idx]
        cur_vel = self.env.vel[self.drone_idx]
        cur_ang_vel = self.env.ang_v[self.drone_idx]
        cur_rpy = self.env.rpy[self.drone_idx]
        
        # Sub-mode execution logic
        if self._mode == "TAKEOFF":
            # Check if takeoff altitude is reached
            alt_diff = abs(cur_pos[2] - self._takeoff_altitude)
            if alt_diff < 0.08:
                if self.verbose:
                    print(f"[GUIDED] Takeoff target of {self._takeoff_altitude:.2f}m reached. Transitioning to loiter.")
                self.mode = "GUIDED"
                self.target_pos = cur_pos.copy()
                self.target_yaw = cur_rpy[2]
                self.target_vel = np.zeros(3)
                
        elif self._mode == "LAND":
            # Ramp target altitude down towards the ground
            self.target_pos[2] = max(0.02, self.target_pos[2] - self._landing_descend_rate * control_timestep)
            self.target_vel = np.array([0.0, 0.0, -self._landing_descend_rate])
            
            # Touchdown detection: altitude close to ground and vertical velocity is near zero
            if cur_pos[2] < 0.06 and abs(cur_vel[2]) < 0.15:
                if self.verbose:
                    print("[GUIDED] Touchdown detected. Automatically disarming drone.")
                self.disarm()
                return np.zeros(4)
                
        elif self._mode == "GUIDED":
            # Integrate body-frame target velocity [vx_body, vy_body, vz_body] and yaw rate setpoints
            if np.linalg.norm(self.target_vel) > 1e-6:
                yaw = cur_rpy[2]
                vx_world = self.target_vel[0] * np.cos(yaw) - self.target_vel[1] * np.sin(yaw)
                vy_world = self.target_vel[0] * np.sin(yaw) + self.target_vel[1] * np.cos(yaw)
                vz_world = self.target_vel[2]
                
                self.target_pos[0] += vx_world * control_timestep
                self.target_pos[1] += vy_world * control_timestep
                self.target_pos[2] += vz_world * control_timestep

            if abs(self.target_yaw_rate) > 1e-6:
                self.target_yaw += self.target_yaw_rate * control_timestep
                self.target_yaw = (self.target_yaw + np.pi) % (2 * np.pi) - np.pi
                
        # Run position and attitude PID loop
        target_vel_world = np.zeros(3)
        if self._mode == "GUIDED":
            # Rotate body-frame command velocity to world coordinates for PID feedforward
            yaw = cur_rpy[2]
            target_vel_world[0] = self.target_vel[0] * np.cos(yaw) - self.target_vel[1] * np.sin(yaw)
            target_vel_world[1] = self.target_vel[0] * np.sin(yaw) + self.target_vel[1] * np.cos(yaw)
            target_vel_world[2] = self.target_vel[2]
        elif self._mode in ["LAND", "TAKEOFF"]:
            # LAND and TAKEOFF target_vel are already in world frame coordinates
            target_vel_world = self.target_vel.copy()

        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=self.target_pos,
            target_rpy=np.array([0.0, 0.0, self.target_yaw]),
            target_vel=target_vel_world
        )
        return rpm.flatten()
