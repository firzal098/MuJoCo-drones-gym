import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from brax.envs.base import Env, State
import numpy as np

from multi_drone_mujoco.utils.enums import DroneModel
from multi_drone_mujoco.envs.base_aviary import _generate_aviary_xml, DRONE_PARAMS

class BaseAviaryJax(Env):
    """
    Base JAX environment for MuJoCo Drones.
    Inherits from Brax's env.Env to be compatible with Brax training pipelines.
    """
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        obstacles: bool = False,
    ):
        super().__init__()
        
        self.drone_model = drone_model
        self.num_drones = num_drones
        self.sim_freq = sim_freq
        self.ctrl_freq = ctrl_freq
        self.sim_steps_per_ctrl = sim_freq // ctrl_freq
        self.dt = 1.0 / ctrl_freq
        
        # Load drone physics parameters
        params = DRONE_PARAMS[drone_model]
        self.mass = params["mass"]
        self.arm_length = params["arm_length"]
        self.kf = params["kf"]
        self.km = params["km"]
        self.gravity = 9.81 * self.mass
        self.max_rpm = jnp.sqrt((params["thrust2weight_ratio"] * self.gravity) / (4 * self.kf))
        self.G = 9.81
        
        # Generate MuJoCo XML
        init_xyzs = np.array([[0.0, 0.0, 0.25]] * num_drones)
        init_rpys = np.zeros((num_drones, 3))
        xml_str = _generate_aviary_xml(
            num_drones, drone_model, init_xyzs, init_rpys, obstacles=obstacles, vision=True
        )
        
        # Instantiate standard MuJoCo model, then convert to MJX
        self.mj_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mj_model.opt.timestep = 1.0 / sim_freq
        self.sys = mjx.put_model(self.mj_model)
        
    @property
    def backend(self):
        return "mjx"
        
    @property
    def action_size(self):
        return 4 * self.num_drones
        
    def reset(self, rng):
        # Override in subclass
        raise NotImplementedError
        
    def step(self, state, action):
        # Override in subclass
        raise NotImplementedError
