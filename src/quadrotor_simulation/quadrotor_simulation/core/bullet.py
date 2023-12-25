from typing import List, Union

from numpy._typing import ArrayLike
import numpy as np
from .physics import QuadrotorPhysics
import pybullet as p
import pybullet_data


DEFAULT_TIME_STEP = 1/240
GRAVITY = 9.81


class QuadrotorPyBullet:
    def __init__(self,
                 # Pybullet parameters
                 physics_server: str = 'GUI',
                 simulation_step: float = DEFAULT_TIME_STEP,
                 render_ground: bool = False,
                 enable_contact: bool = True,
                 # Quadrotor parameters
                 quadrotor_descriptions: Union[List[str], None] = None,
                 quadrotor_initial_poses: Union[List[ArrayLike], None] = None,
                 quadrotor_initial_twists: Union[List[ArrayLike], None] = None,
                 quadrotor_initial_rotor_speeds: Union[List[ArrayLike], None] = None,
                 quadrotors_parameters: Union[List[dict], None] = None,
                 # Quadrotor physics parameters
                 enable_rotor_dynamics: bool = False,
                 enable_rotor_drag: bool = False,
                 enable_fuselage_drag: bool = False,
                 # Obstacles parameters
                 obstacle_descriptions: Union[List[str], None] = None,
                 obstacle_poses: Union[List[ArrayLike], None] = None,
                 ):
        """
        Initialize a quadrotor simulation using PyBullet.

        Args:
            physics_server: 'GUI' or 'DIRECT'
            simulation_step: simulation time step in seconds, default is 1/240 as with Pybullet
            render_ground: whether to render the ground plane
            enable_contact: whether to enable contact between objects
            quadrotor_descriptions: list of quadrotor descriptions in URDF format
            quadrotor_initial_poses: list of quadrotor initial poses (position, orientation in quaternion xyzw format)
            quadrotor_initial_twists: list of quadrotor initial twists (linear velocity, angular velocity)
            quadrotor_initial_rotor_speeds: list of quadrotor initial rotor speeds (rad/s)
            quadrotor_parameters: list of quadrotor parametersameters (see QuadrotorPhysics) for each quadrotor.
            enable_rotor_dynamics: whether to enable rotor dynamics
            enable_rotor_drag: whether to enable rotor rotor drag
            enable_fuselage_drag: whether to enable fuselage drag
            obstacle_descriptions: list of obstacle descriptions in URDF format
            obstacle_poses: list of obstacle initial poses (position, orientation in quaternion xyzw format)
        """
        pass

    def init_pybullet(self,
                      physics_server: str,
                      simulation_step: float,
                      render_ground: bool,
                      enable_contact: bool):
        """
        Initialize PyBullet simulation.
        """
        self._physics_client_id = p.connect(p.GUI if physics_server == 'GUI' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(simulation_step)

    def __del__(self):
        """
        Close PyBullet simulation.
        """
        p.disconnect(self._physics_client_id)
