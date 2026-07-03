"""Example: Replicate KRTI 2026 Webots arena in MuJoCo.

This script parses and replicates the gates, waypoints, landing pads,
and other structures from krti2026 camera.wbt into the MuJoCo environment.
It then runs a PID course controller to guide the drone through the course.
"""

import os
# === WSL GPU Rendering Fix ===
# Force Mesa to use the D3D12 backend (routes OpenGL through NVIDIA GPU via DirectX)
os.environ["GALLIUM_DRIVER"] = "d3d12"
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
# Disable V-Sync to prevent frame rate capping and stutter under WSLg
os.environ["vblank_mode"] = "0"
os.environ["__GL_SYNC_TO_VBLANK"] = "0"
# Force EGL backend for off-screen (headless) camera rendering — avoids GLFW overhead
os.environ.setdefault("MUJOCO_GL", "egl")

import time
import threading
import queue
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from multi_drone_mujoco.envs.base_aviary import BaseAviary, _generate_aviary_xml
from multi_drone_mujoco.control.guided_mode import GuidedVehicle
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType

def make_wp(name, x, y):
    return f"""
    <!-- Waypoint: {name} -->
    <body name="{name}" pos="{x} {y} 0.05">
      <geom type="box" size="1.0 1.0 0.05" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.25 0.25 0.025" pos="0 0 0.05" rgba="1 1 1 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_red_box(x, y):
    return f"""
    <!-- Red Box Waypoint -->
    <body name="wp_red_box" pos="{x} {y} 0.05">
      <!-- Base Pad -->
      <geom type="box" size="1.0 1.0 0.05" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <!-- Red Box Bottom -->
      <geom type="box" size="0.31 0.2075 0.0125" pos="0 0 0.0625" rgba="1 0 0 1" contype="1" conaffinity="1"/>
      <!-- Red Box Walls -->
      <geom type="box" size="0.005 0.2075 0.0725" pos="0.3 0 0.1475" rgba="1 0 0 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.005 0.2075 0.0725" pos="-0.3 0 0.1475" rgba="1 0 0 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.005 0.31 0.0725" pos="0 -0.2 0.1475" euler="0 0 90" rgba="1 0 0 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.005 0.31 0.0725" pos="0 0.2 0.1475" euler="0 0 90" rgba="1 0 0 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_line_pads(x, y, yaw_deg):
    pads_xml = ""
    y_offsets = [0, -2.6, -5.19, -7.72, -10.27]
    for idx, dy in enumerate(y_offsets):
        pads_xml += f'      <geom type="box" size="0.225 1.0 0.05" pos="0 {dy} 0.05" rgba="0 0 0 1" contype="1" conaffinity="1"/>\n'
    return f"""
    <!-- Black Line Pads -->
    <body name="black_line_pads" pos="{x} {y} 0" euler="0 0 {yaw_deg}">
{pads_xml}    </body>
    """

def make_start_zone(x, y):
    return f"""
    <!-- Start Zone -->
    <body name="start_zone" pos="{x} {y} 0.05">
      <geom type="box" size="0.5 0.5 0.05" rgba="0 0 1 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_landing_pad(x, y, z):
    return f"""
    <!-- Landing Pad -->
    <body name="landing_pad" pos="{x} {y} {z}">
      <geom type="box" size="0.75 0.5 0.05" rgba="0 0 1 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_elevation(x, y, z):
    return f"""
    <!-- Elevation Landscape -->
    <body name="elevation_cylinder" pos="{x} {y} {z}">
      <geom type="cylinder" size="50.0 0.5" rgba="0.435 0.647 0.369 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_single_gate_xml(name, x, y, z, yaw_deg=0.0):
    return f"""
    <!-- Single Gate: {name} -->
    <body name="{name}" pos="{x} {y} {z}" euler="0 0 {yaw_deg}">
      <!-- Vertical Poles -->
      <geom type="cylinder" size="0.007 0.99" pos="0 0 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.99" pos="1.90312 0 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <!-- Horizontal Pole -->
      <geom type="cylinder" size="0.007 0.95156" pos="0.95312 0 1.99" euler="0 90 0" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <!-- Banners -->
      <geom type="box" size="0.10801 0.002 0.75" pos="1.79511 0 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="0.10511 0 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.95156" pos="0.95511 0 1.88" euler="0 90 0" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_double_gate_xml(name, x, y, z, yaw_deg=0.0):
    return f"""
    <!-- Double Gate: {name} -->
    <body name="{name}" pos="{x} {y} {z}" euler="0 0 {yaw_deg}">
      <!-- Gate 1 (y = 0) -->
      <geom type="cylinder" size="0.007 0.99" pos="0 0 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.99" pos="1.90312 0 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.95156" pos="0.95312 0 1.99" euler="0 90 0" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="1.79511 0 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="0.10511 0 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.95156" pos="0.95511 0 1.88" euler="0 90 0" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>

      <!-- Gate 2 (y = 1) -->
      <geom type="cylinder" size="0.007 0.99" pos="0 1 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.99" pos="1.90312 1 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.95156" pos="0.95312 1 1.99" euler="0 90 0" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="1.79511 1 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="0.10511 1 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.95156" pos="0.95511 1 1.88" euler="0 90 0" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>

      <!-- Connecting Banners (at y = 0.5) -->
      <geom type="box" size="0.5 0.002 0.85801" pos="-0.004888 0.5 1.13" euler="0 0 90" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.5 0.002 0.85801" pos="1.89511 0.5 1.13" euler="0 0 90" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.5 0.002 0.95" pos="0.94511 0.5 1.99" axisangle="0.5773489358556708 0.5773509358554485 -0.5773509358554485 -120" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
    </body>
    """

def make_triple_gate_xml(name, x, y, z, yaw_deg=0.0):
    return f"""
    <!-- Triple Gate: {name} -->
    <body name="{name}" pos="{x} {y} {z}" euler="0 0 {yaw_deg}">
      <!-- Gate 1 (y = 0) -->
      <geom type="cylinder" size="0.007 0.99" pos="0 0 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.99" pos="1.90312 0 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.95156" pos="0.95312 0 1.99" euler="0 90 0" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="1.79511 0 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="0.10511 0 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.95156" pos="0.95511 0 1.88" euler="0 90 0" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>

      <!-- Gate 2 (y = 1) -->
      <geom type="cylinder" size="0.007 0.99" pos="0 1 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.99" pos="1.90312 1 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.95156" pos="0.95312 1 1.99" euler="0 90 0" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="1.79511 1 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="0.10511 1 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.95156" pos="0.95511 1 1.88" euler="0 90 0" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>

      <!-- Connecting Banners 1 (at y = 0.5) -->
      <geom type="box" size="0.5 0.002 0.85801" pos="-0.004888 0.5 1.13" euler="0 0 90" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.5 0.002 0.85801" pos="1.89511 0.5 1.13" euler="0 0 90" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.5 0.002 0.95" pos="0.94511 0.5 1.99" axisangle="0.5773489358556708 0.5773509358554485 -0.5773509358554485 -120" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>

      <!-- Gate 3 (y = 2) -->
      <geom type="cylinder" size="0.007 0.99" pos="0 2 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.99" pos="1.90312 2 0.99" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="cylinder" size="0.007 0.95156" pos="0.95312 2 1.99" euler="0 90 0" rgba="0.2 0.2 0.2 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="1.79511 2 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.75" pos="0.10511 2 1.02" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.10801 0.002 0.95156" pos="0.95511 2 1.88" euler="0 90 0" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>

      <!-- Connecting Banners 2 (at y = 1.5) -->
      <geom type="box" size="0.5 0.002 0.85801" pos="-0.004888 1.5 1.13" euler="0 0 90" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.5 0.002 0.85801" pos="1.89511 1.5 1.13" euler="0 0 90" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
      <geom type="box" size="0.5 0.002 0.95" pos="0.94511 1.5 1.99" axisangle="0.5773489358556708 0.5773509358554485 -0.5773509358554485 -120" rgba="0.91 0.57 0.07 1" contype="1" conaffinity="1"/>
    </body>
    """

def generate_krti_arena_xml():
    xml = ""
    # 1. Waypoints (WP1)
    xml += make_wp("wp1_a", -5.64, 22.03)
    xml += make_wp("wp1_b", -12.11, 13.34)
    xml += make_wp("wp1_c", -18.37, -0.78)
    
    # 2. Red Box Waypoint
    xml += make_red_box(-12.14, 22.04)
    
    # 3. Black Line Pads
    xml += make_line_pads(-12.661, 10.7874, -30.0)
    
    # 4. StartZone (global: [0.88, 24.45, 0.05])
    xml += make_start_zone(0.88, 24.45)
    
    # 5. LandingPad
    xml += make_landing_pad(0.07, -13.69, 0.67)
    
    # 6. Elevation Landscape (large green cylinder)
    xml += make_elevation(0.31, -62.04, 0.19)
    
    # 7. Single Gates
    xml += make_single_gate_xml("gate_single_a", 0.17, 9.26, 0.0, yaw_deg=0.0)
    xml += make_single_gate_xml("gate_single_b", -19.31, -2.88, 0.0, yaw_deg=0.0)
    xml += make_single_gate_xml("gate_single_c", -2.93, 14.59, 0.0, yaw_deg=0.0)
    
    # 8. Double Gate
    xml += make_double_gate_xml("gate_double_a", -8.33, 21.1, 0.0, yaw_deg=90.0)
    
    # 9. Triple Gate
    xml += make_triple_gate_xml("gate_triple_a", -13.05, 16.6, 0.0, yaw_deg=0.0)
    
    return xml

class KRTIAviary(BaseAviary):
    """Subclass of BaseAviary containing the replicated KRTI 2026 Arena."""
    
    def __init__(self, **kwargs):
        # Override DRONE_PARAMS at runtime before initialization so MuJoCo compiles
        # the model with matching mass/inertia/geometry properties.
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

        kwargs["num_drones"] = kwargs.get("num_drones", 1)
        if "initial_xyzs" not in kwargs:
            # Drone starts on the start zone (z elevated to 0.25 to prevent collision penetration at startup)
            kwargs["initial_xyzs"] = np.array([[0.92, 24.47, 0.25]])
        
        super().__init__(**kwargs)
        
        # Override image resolution to be higher for better front camera visualization
        if self.VISION_ATTR:
            self.IMG_RES = np.array([320, 240])
            self.rgb = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
            self.dep = np.ones((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.float32)
            self.seg = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.int32)
        
        # 1. Regenerate standard aviary XML
        base_xml = _generate_aviary_xml(
            num_drones=self.NUM_DRONES,
            drone_model=self.DRONE_MODEL,
            init_xyzs=self.INIT_XYZS,
            init_rpys=self.INIT_RPYS,
            obstacles=False, # Disable standard block/sphere/cylinder obstacles
            vision=self.VISION_ATTR,
            timestep=self.SIM_TIMESTEP,
        )
        
        # 2. Re-scale ground floor plane to 80m x 80m to contain the whole arena
        base_xml = base_xml.replace('size="10 10 0.05"', 'size="80 80 0.05"')
        
        # 3. Generate our custom KRTI arena XML elements
        arena_xml = generate_krti_arena_xml()
        
        # 4. Insert arena bodies into the worldbody
        insert_idx = base_xml.find("</worldbody>")
        if insert_idx == -1:
            raise ValueError("Could not find </worldbody> in generated MuJoCo XML")
            
        krti_xml = base_xml[:insert_idx] + arena_xml + base_xml[insert_idx:]
        
        # 5. Load model and data from the modified XML string
        self.model = mujoco.MjModel.from_xml_string(krti_xml)
        self.data = mujoco.MjData(self.model)

        # ======================================================================
        # 6. Override physics to match KRTIDrone Webots PROTO
        # ======================================================================
        # Source values extracted from KRTIDrone.proto:
        #   physics Physics { mass 3.0  inertiaMatrix [0.08 0.08 0.15  0 0 0] }
        #   thrustConstants 0.002 0   → kf = 0.002  N/(rad/s)²
        #   torqueConstants 8.0e-04 0 → km = 8.0e-4 N·m/(rad/s)²
        #   maxVelocity 100           → max motor speed = 100 rad/s
        #   boundingObject Box { size 0.470 0.470 0.110 }
        #   Motor centerOfThrust positions: (±0.130, ±0.200, 0.023)

        _IRIS_MASS    = 3.000                  # kg
        _IRIS_IXX     = 0.080                  # kg·m²
        _IRIS_IYY     = 0.080                  # kg·m²
        _IRIS_IZZ     = 0.150                  # kg·m²
        _IRIS_KF      = 0.002                  # N/(rad/s)²
        _IRIS_KM      = 8.0e-4                 # N·m/(rad/s)²
        _IRIS_MAX_RAD = 100.0                  # rad/s  (maxVelocity from PROTO)
        # Arm length = distance from body centre to motor = sqrt(0.130²+0.200²)
        _IRIS_L       = np.sqrt(0.130**2 + 0.200**2)   # ≈ 0.2386 m

        # -- 6a. Patch MuJoCo body inertial properties --
        for d in range(self.NUM_DRONES):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{d}"
            )
            if body_id >= 0:
                self.model.body_mass[body_id]    = _IRIS_MASS
                self.model.body_inertia[body_id] = np.array([_IRIS_IXX, _IRIS_IYY, _IRIS_IZZ])

        # -- 6b. Patch propeller site positions to match Iris motor layout --
        # Reordered to match the index mapping assumed by base_aviary's torque equations:
        #   prop0: front-left (CW/m3 in Webots)   → ( 0.130,  0.200, 0.023)
        #   prop1: front-right (CCW/m1 in Webots) → ( 0.130, -0.200, 0.023)
        #   prop2: rear-right (CW/m4 in Webots)   → (-0.130, -0.200, 0.023)
        #   prop3: rear-left (CCW/m2 in Webots)  → (-0.130,  0.200, 0.023)
        _iris_prop_offsets = [
            ( 0.130,  0.200, 0.023),   # prop0 – front-left
            ( 0.130, -0.200, 0.023),   # prop1 – front-right
            (-0.130, -0.200, 0.023),   # prop2 – rear-right
            (-0.130,  0.200, 0.023),   # prop3 – rear-left
        ]
        for d in range(self.NUM_DRONES):
            for i, (px, py, pz) in enumerate(_iris_prop_offsets):
                site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, f"drone{d}_prop{i}"
                )
                if site_id >= 0:
                    self.model.site_pos[site_id] = np.array([px, py, pz])

        # -- 6c. Patch collision cylinder size --
        # Webots boundingObject Box { size 0.470 0.470 0.110 }
        # → cylinder approximation: radius = 0.470/2, half-height = 0.110/2
        for d in range(self.NUM_DRONES):
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"drone{d}_collision"
            )
            if geom_id >= 0:
                self.model.geom_size[geom_id] = np.array([0.235, 0.055, 0.0])

        # -- 6d. Re-sync all Python-side derived constants --
        # (PIDControl reads these from env at construction time — must be set before ctrl = PIDControl(env))
        self.M                   = _IRIS_MASS
        self.L                   = _IRIS_L
        self.KF                  = _IRIS_KF
        self.KM                  = _IRIS_KM
        self.J                   = np.diag([_IRIS_IXX, _IRIS_IYY, _IRIS_IZZ])
        self.J_INV               = np.linalg.inv(self.J)
        self.GRAVITY             = self.G * self.M                                       # 29.43 N
        self.HOVER_RPM           = np.sqrt(self.GRAVITY / (4.0 * self.KF))              # ≈ 60.6 rad/s
        self.MAX_RPM             = _IRIS_MAX_RAD                                         # 100 rad/s (from maxVelocity)
        self.THRUST2WEIGHT_RATIO = (4.0 * self.KF * self.MAX_RPM**2) / self.GRAVITY     # ≈ 2.72
        self.MAX_THRUST          = 4.0 * self.KF * self.MAX_RPM**2
        self.MAX_XY_TORQUE       = (2.0 * self.L * self.KF * self.MAX_RPM**2) / np.sqrt(2)
        self.MAX_Z_TORQUE        = 2.0 * self.KM * self.MAX_RPM**2
        self.COLLISION_R         = 0.235
        self.COLLISION_H         = 0.110
        self.PROP_RADIUS         = 0.127   # ≈ 10" props standard on 3DR Iris

        print(f"[KRTIAviary] Applied Iris physics: mass={self.M:.3f} kg  "
              f"hover_rpm={self.HOVER_RPM:.1f} rad/s  max_rpm={self.MAX_RPM:.0f} rad/s  "
              f"T/W={self.THRUST2WEIGHT_RATIO:.2f}")



def get_gate_corners(gate_type):
    """Return 3D bounding corners for the gate OPENING (inner aperture).
    
    Using z=[0.2, 1.85] avoids the ground-level z=0 corners that cause the
    bounding box to explode in size when the drone is close and looking up.
    dx is slightly inset from the poles so the box wraps the passable opening.
    """
    # Gate inner width (slightly inset from pole edges)
    x_min, x_max = 0.07, 1.83
    # Gate inner height (above ground, below horizontal bar)
    z_min, z_max = 0.20, 1.85
    # Small y-thickness so the gate has 3D extent (prevents degenerate projection)
    t = 0.08

    if gate_type == "single":
        y_extents = [-t, t]
    elif gate_type == "double":
        y_extents = [-t, 1.0 + t]
    elif gate_type == "triple":
        y_extents = [-t, 2.0 + t]
    else:
        y_extents = [-t, t]

    corners = []
    for dx in [x_min, x_max]:
        for dy in y_extents:
            for dz in [z_min, z_max]:
                corners.append(np.array([dx, dy, dz]))
    return corners

def project_point(p_world, cam_pos, cam_mat, fovy, width, height):
    R = cam_mat.reshape(3, 3)
    dp = p_world - cam_pos
    p_cam = R.T @ dp
    x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]
    
    if z_c >= 0:
        return None
        
    depth = -z_c
    f_y = 1.0 / np.tan(np.deg2rad(fovy) / 2.0)
    f_x = f_y * (height / width)
    
    ndc_x = f_x * (x_c / depth)
    ndc_y = f_y * (y_c / depth)
    
    px_x = (ndc_x + 1.0) / 2.0 * width
    px_y = (1.0 - ndc_y) / 2.0 * height
    
    return px_x, px_y

def get_text_size(draw, text, font):
    if font is None:
        return len(text) * 6, 12
    try:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        elif hasattr(draw, "textbbox"):
            # Can raise ValueError for non-TrueType default bitmap fonts
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        elif hasattr(font, "getsize"):
            return font.getsize(text)
    except Exception:
        pass
    return len(text) * 6, 12

def draw_fake_yolo_boxes(rgb_img, model, data, cam_id, gates, cam_width, cam_height):
    """Draw projected bounding boxes for gates.
    
    Takes explicit model/data instead of env so the background camera thread
    can pass its own data snapshot — avoiding race conditions with the main thread.
    """
    draw = ImageDraw.Draw(rgb_img)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].copy()
    fovy = model.cam_fovy[cam_id]

    for gate_name, gate_type in gates.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gate_name)
        if body_id < 0:
            continue

        T_gate = data.xpos[body_id].copy()
        R_gate = data.xmat[body_id].copy().reshape(3, 3)

        local_corners = get_gate_corners(gate_type)

        px_list = []
        for pt_local in local_corners:
            pt_world = R_gate @ pt_local + T_gate
            px = project_point(pt_world, cam_pos, cam_mat, fovy, cam_width, cam_height)
            if px is not None:
                px_list.append(px)

        if len(px_list) < 2:
            # Not enough visible corners (e.g., drone is inside the gate) — skip
            continue

        xs = [p[0] for p in px_list]
        ys = [p[1] for p in px_list]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Clamp to image bounds so a nearby gate never produces an off-screen box
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        x_max = min(float(cam_width - 1), x_max)
        y_max = min(float(cam_height - 1), y_max)

        # Only draw if box has a meaningful size on screen
        if x_max - x_min < 4 or y_max - y_min < 4:
            continue
        if x_max < 0 or x_min >= cam_width or y_max < 0 or y_min >= cam_height:
            continue

        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)

        label = f"{gate_name} 0.95"
        tw, th = get_text_size(draw, label, font)
        text_y = max(0, y_min - th - 4)
        draw.rectangle([x_min, text_y, x_min + tw + 4, text_y + th + 4], fill=(0, 255, 0))
        draw.text((x_min + 2, text_y + 2), label, fill=(0, 0, 0), font=font)


def main():
    import sys
    # ============================================================
    # CONFIGURATION SWITCHES
    # ============================================================
    # True = run with graphical visualizer window (WSLg)
    # False = run headless (much faster, exits after 1 full lap)
    USE_GUI = True
    # True = show a live pygame window with the drone's front camera feed and YOLO bounding boxes
    SHOW_FRONT_CAM = True

    if "--headless" in sys.argv:
        USE_GUI = False
    if "--no-front-cam" in sys.argv:
        SHOW_FRONT_CAM = False
    # ============================================================

    print("=" * 60)
    print("KRTI 2026 Replicated Arena Simulator")
    print(f"  Mode: {'GUI Visualizer' if USE_GUI else 'Headless'}")
    print("  Drone course traversal automatically using PID control.")
    print("=" * 60)

    # Initialize custom KRTIAviary with vision attributes enabled
    env = KRTIAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        act_type=ActionType.RPM,
        gui=USE_GUI,
        vision_attributes=True,
        render_mode="human" if USE_GUI else None
    )

    # Initialize Guided mode wrapper
    vehicle = GuidedVehicle(env)
    vehicle.arm()

    obs, info = env.reset()

    # Gate names and types dictionary for YOLO bounding box detection
    gates = {
        "gate_single_a": "single",
        "gate_single_b": "single",
        "gate_single_c": "single",
        "gate_double_a": "double",
        "gate_triple_a": "triple",
    }

    # Initialize Pygame for live front camera display
    pygame_ok = False
    display_scale = 2
    cam_width, cam_height = 320, 240
    if SHOW_FRONT_CAM:
        try:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((cam_width * display_scale, cam_height * display_scale))
            pygame.display.set_caption("Drone Front Camera - Fake YOLO Gate Detection")
            clock = pygame.time.Clock()
            pygame_ok = True
            print("Successfully initialized live front camera display window.")
        except Exception as e:
            print(f"Could not initialize Pygame window: {e}. Running in headless/save-only mode.")

    # Find the camera ID for drone0_cam
    cam_name = "drone0_cam"
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        print("Warning: drone0_cam not found in environment!")

    # Video recording container
    frames = []

    # --- Background camera thread setup ---
    # A shared queue: the sim loop posts (model, data) snapshots; the camera thread renders them.
    _cam_frame_queue: queue.Queue = queue.Queue(maxsize=2)  # maxsize=2 → never stalls sim
    _latest_frame = {"rgb": None, "lock": threading.Lock()}  # latest rendered frame for pygame
    _cam_renderer = None  # created inside the thread

    def _camera_thread_fn():
        """Background thread: renders front-camera frames at up to 60 FPS without blocking the sim."""
        nonlocal _cam_renderer, frames
        local_renderer = mujoco.Renderer(env.model, height=cam_height, width=cam_width)
        local_data = mujoco.MjData(env.model)
        target_dt = 1.0 / 60.0  # target 60 FPS for the camera

        while True:
            t0 = time.perf_counter()
            try:
                data_snapshot = _cam_frame_queue.get(timeout=0.5)
            except queue.Empty:
                if not threading.current_thread().daemon:
                    break
                continue
            if data_snapshot is None:  # sentinel → exit
                break

            # Copy state into local data
            mujoco.mj_copyData(local_data, env.model, data_snapshot)

            # Render RGB frame
            local_renderer.update_scene(local_data, camera=cam_name)
            rgb = local_renderer.render()

            img = Image.fromarray(rgb)
            draw_fake_yolo_boxes(img, env.model, local_data, cam_id, gates, cam_width, cam_height)
            frames.append(np.array(img))

            with _latest_frame["lock"]:
                _latest_frame["rgb"] = np.array(img)  # for pygame display

            # Pace to target FPS
            elapsed = time.perf_counter() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    cam_thread = threading.Thread(target=_camera_thread_fn, daemon=True)
    cam_thread.start()

    # Initial target yaw (pointing forward / zero)
    target_yaw = env.rpy[0, 2]

    # Pre-defined course waypoints matching replicated gates and pads
    waypoints = [
        # Visit index: [X, Y, Z], Name/Description
        (np.array([0.92, 24.47, 1.0]), "Hover above Start Zone"),
        (np.array([1.12, 9.26, 1.0]), "Single Gate A (Fly-through)"),
        (np.array([-5.64, 22.03, 1.0]), "Waypoint 1A (Hover/Flyover)"),
        (np.array([-8.83, 22.05, 1.0]), "Double Gate (Tunnel Entrance)"),
        (np.array([-9.83, 22.05, 1.0]), "Double Gate (Tunnel Exit)"),
        (np.array([-12.14, 22.04, 1.0]), "Red Box (Hover/Flyover)"),
        (np.array([-12.1, 16.6, 1.0]), "Triple Gate (Tunnel Pass)"),
        (np.array([-12.11, 13.34, 1.0]), "Waypoint 1B (Hover/Flyover)"),
        (np.array([-12.661, 10.7874, 1.0]), "Black Line Pad 1"),
        (np.array([-18.36, -2.88, 1.0]), "Single Gate B (Fly-through)"),
        (np.array([-18.37, -0.78, 1.0]), "Waypoint 1C (Hover/Flyover)"),
        (np.array([0.07, -13.69, 1.2]), "Landing Pad (Hover/Land)"),
    ]
    
    wp_idx = 0
    target_pos = waypoints[wp_idx][0]
    wp_name = waypoints[wp_idx][1]
    course_completed = False

    # Command initial guided mode target
    vehicle.simple_goto(target_pos, target_yaw)

    # Call render once to spawn the viewer
    if USE_GUI:
        env.render()

    print(f"\nSimulation started. Course: {len(waypoints)} steps.")
    print(f"Moving to Waypoint {wp_idx + 1}/{len(waypoints)}: {wp_name} at {target_pos}")

    step = 0
    last_render_time = 0.0
    
    # FPS tracking variables
    last_fps_time = time.time()
    loop_step_count = 0
    render_count = 0
    
    running = True
    while running:
        if USE_GUI:
            # Check if GUI window was closed
            if env._viewer is None or not env._viewer.is_running():
                running = False
                break
        else:
            # In headless mode, stop after 1 full course completion
            if course_completed:
                print("\nTrajectory course completed successfully in headless mode!")
                running = False
                break

        start_time = time.time()
        loop_step_count += 1

        # Check distance to current waypoint target
        dist = np.linalg.norm(env.pos[0] - target_pos)
        
        # If close to current waypoint, advance to next
        if dist < 0.15:
            print(f"  [SUCCESS] Reached Waypoint {wp_idx + 1}: {wp_name}!")
            wp_idx += 1
            if wp_idx >= len(waypoints):
                print("  [FINISH] Trajectory course completed successfully! Restarting course...")
                wp_idx = 0
                course_completed = True
            
            target_pos = waypoints[wp_idx][0]
            wp_name = waypoints[wp_idx][1]
            print(f"  Target set: Waypoint {wp_idx + 1}/{len(waypoints)}: {wp_name} at {target_pos}")

        # Align yaw dynamically to face the next target waypoint
        dist_xy = np.linalg.norm(target_pos[:2] - env.pos[0, :2])
        if dist_xy > 0.3:
            target_yaw = np.arctan2(target_pos[1] - env.pos[0, 1], target_pos[0] - env.pos[0, 0])

        # Update target on our guided vehicle wrapper
        vehicle.simple_goto(target_pos, target_yaw)

        # Compute guided vehicle control inputs (motor RPMs)
        rpm = vehicle.update(control_timestep=env.CTRL_TIMESTEP)

        if step < 20:
            print(f"Step {step}: pos={env.pos[0]}, quat={env.quat[0]}, vel={env.vel[0]}, ang_vel={env.ang_v[0]}, rpm={rpm}")

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(rpm)

        # Push a lightweight data snapshot to the camera thread (non-blocking)
        if env.VISION_ATTR:
            # Create a copy of MjData for the background thread (only if queue has space)
            if not _cam_frame_queue.full():
                data_copy = mujoco.MjData(env.model)
                mujoco.mj_copyData(data_copy, env.model, env.data)
                _cam_frame_queue.put_nowait(data_copy)

        # Display the latest rendered frame in pygame (non-blocking, uses pre-rendered frame)
        if pygame_ok:
            with _latest_frame["lock"]:
                frame_rgb = _latest_frame["rgb"]
            if frame_rgb is not None:
                # Zero-copy path: surfarray is faster than image.fromstring
                surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                surf_scaled = pygame.transform.scale(surf, (cam_width * display_scale, cam_height * display_scale))
                screen.blit(surf_scaled, (0, 0))
                pygame.display.flip()
            # Check for Pygame quit event to exit cleanly
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Render at most 30 FPS to reduce rendering overhead and lag under WSLg
        current_time = time.time()
        if USE_GUI and current_time - last_render_time >= 1.0 / 30.0:
            env.render()
            last_render_time = current_time
            render_count += 1

        # Periodic telemetry print with Loop Rate & Render FPS
        if step == 0:
            print(f"t=  0.0s | pos=[{env.pos[0,0]:+.2f}, {env.pos[0,1]:+.2f}, {env.pos[0,2]:+.2f}] | target_err={dist:.4f}")
        elif step % 48 == 0:
            now = time.time()
            elapsed_fps = now - last_fps_time
            current_loop_rate = loop_step_count / elapsed_fps
            current_render_fps = render_count / elapsed_fps
            
            # Reset counters
            loop_step_count = 0
            render_count = 0
            last_fps_time = now
            
            fps_str = f" | Loop Rate={current_loop_rate:.1f}Hz"
            if USE_GUI:
                fps_str += f" | Render FPS={current_render_fps:.1f}"
            print(f"t={step * env.CTRL_TIMESTEP:5.1f}s | pos=[{env.pos[0,0]:+.2f}, {env.pos[0,1]:+.2f}, {env.pos[0,2]:+.2f}] | target_err={dist:.4f}{fps_str}")

        step += 1

        # Reset if crashed/out-of-bounds (excluding landscape cylinder bounds)
        rpy = env.rpy[0]
        has_crashed = env.pos[0, 2] < 0.05 or env.pos[0, 2] > 5.0 or abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2

        if has_crashed:
            print("  [CRASH] Drone crashed or went out of bounds! Resetting course...")
            obs, info = env.reset()
            vehicle.disarm()
            vehicle.arm()
            wp_idx = 0
            target_pos = waypoints[wp_idx][0]
            wp_name = waypoints[wp_idx][1]
            target_yaw = env.rpy[0, 2]
            vehicle.simple_goto(target_pos, target_yaw)

        # Pace loop to real-time
        elapsed = time.time() - start_time
        if elapsed < env.CTRL_TIMESTEP:
            time.sleep(env.CTRL_TIMESTEP - elapsed)

    # Signal camera thread to stop and wait for it
    _cam_frame_queue.put(None)
    cam_thread.join(timeout=3.0)

    env.close()

    # Save the recorded camera frames
    if len(frames) > 0:
        os.makedirs("/home/firza/MuJoCo-drones-gym/multi_drone_mujoco/results", exist_ok=True)
        gif_path = "/home/firza/MuJoCo-drones-gym/multi_drone_mujoco/results/front_cam_yolo.gif"
        print(f"\nSaving front camera recording to {gif_path}...")
        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=40, # ~24 fps
                loop=0
            )
            print("Successfully saved GIF recording!")
        except Exception as e:
            print(f"Could not save recording as GIF: {e}")

        # Also try to save as MP4 if imageio is available
        video_path = "/home/firza/MuJoCo-drones-gym/multi_drone_mujoco/results/front_cam_yolo.mp4"
        try:
            import imageio
            print(f"Attempting to save MP4 recording to {video_path}...")
            imageio.mimsave(video_path, frames, fps=24)
            print("Successfully saved MP4 recording!")
        except Exception as e:
            print(f"Could not save MP4 recording: {e}")
    print("\nViewer closed. Exit successful.")

if __name__ == "__main__":
    main()
