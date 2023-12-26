from typing import List, Union

import numpy as np
from numpy._typing import ArrayLike
import pybullet_data
from scipy.spatial.transform import Rotation

import pybullet as p

from .physics import QuadrotorPhysics


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
            quadrotor_descriptions: list of paths for quadrotor descriptions in URDF format
            quadrotor_initial_poses: list of quadrotor initial poses (position, orientation in quaternion xyzw format)
            quadrotor_initial_twists: list of quadrotor initial twists (linear velocity, angular velocity)
            quadrotor_initial_rotor_speeds: list of quadrotor initial rotor speeds (rad/s)
            quadrotor_parameters: list of quadrotor parameters (see QuadrotorPhysics) for each quadrotor.
            enable_rotor_dynamics: whether to enable rotor dynamics
            enable_rotor_drag: whether to enable rotor rotor drag
            enable_fuselage_drag: whether to enable fuselage drag
            obstacle_descriptions: list of paths for obstacle descriptions in URDF format
            obstacle_poses: list of obstacle initial poses (position, orientation in quaternion xyzw format)
        """
        self.quadrotor_bullet_ids: list[int] = []
        self.obstacle_bullet_ids: list[int] = []
        self.quadrotor_physics_objects: list[QuadrotorPhysics] = []

        self.init_pybullet(physics_server=physics_server,
                           simulation_step=simulation_step,
                           render_ground=render_ground,
                           enable_contact=enable_contact,
                           )
        if quadrotor_descriptions is not None:
            if quadrotor_initial_poses is None:
                quadrotor_initial_poses = [(0, 0, 0, 0, 0, 0, 1) for _ in range(len(quadrotor_descriptions))]
            if quadrotor_initial_twists is None:
                quadrotor_initial_twists = [(0, 0, 0, 0, 0, 0) for _ in range(len(quadrotor_descriptions))]
            if quadrotor_initial_rotor_speeds is None:
                quadrotor_initial_rotor_speeds = [(0, 0, 0, 0) for _ in range(len(quadrotor_descriptions))]
            if quadrotors_parameters is None:
                raise ValueError('quadrotors_parameters must be specified if quadrotor_descriptions is specified')
            self.init_quadrotors(quadrotor_descriptions=quadrotor_descriptions,
                                 quadrotor_initial_poses=quadrotor_initial_poses,
                                 quadrotor_initial_twists=quadrotor_initial_twists,
                                 quadrotor_initial_rotor_speeds=quadrotor_initial_rotor_speeds,
                                 quadrotors_parameters=quadrotors_parameters,
                                 enable_rotor_dynamics=enable_rotor_dynamics,
                                 enable_rotor_drag=enable_rotor_drag,
                                 enable_fuselage_drag=enable_fuselage_drag,
                                 simulation_step=simulation_step,
                                 )
        if obstacle_descriptions is not None:
            if obstacle_poses is None:
                obstacle_poses = [(0, 0, 0, 0, 0, 0, 1) for _ in range(len(obstacle_descriptions))]
            self.init_obstacles(obstacle_descriptions=obstacle_descriptions,
                                obstacle_poses=obstacle_poses,
                                )

    @ property
    def num_quadrotos(self):
        return len(self.quadrotor_bullet_ids)

    def init_pybullet(self,
                      physics_server: str,
                      simulation_step: float,
                      render_ground: bool,
                      enable_contact: bool,
                      ):
        """
        Initialize PyBullet simulation.
        """
        self._physics_client_id = p.connect(p.GUI if physics_server == 'GUI' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -GRAVITY, physicsClientId=self._physics_client_id)
        p.setTimeStep(simulation_step, physicsClientId=self._physics_client_id)
        if render_ground:
            self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        if not enable_contact:
            pass  # TODO: implement disable contact
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self._physics_client_id)

    def init_quadrotors(self,
                        quadrotor_descriptions: List[str],
                        quadrotor_initial_poses: List[ArrayLike],
                        quadrotor_initial_twists: List[ArrayLike],
                        quadrotor_initial_rotor_speeds: List[ArrayLike],
                        quadrotors_parameters: List[dict],
                        enable_rotor_dynamics: bool,
                        enable_rotor_drag: bool,
                        enable_fuselage_drag: bool,
                        simulation_step: float,
                        ):

        for i in range(len(quadrotor_descriptions)):
            self.add_quadrotor(quadrotor_description=quadrotor_descriptions[i],
                               quadrotor_initial_pose=quadrotor_initial_poses[i],
                               quadrotor_initial_twist=quadrotor_initial_twists[i],
                               quadrotor_initial_rotor_speed=quadrotor_initial_rotor_speeds[i],
                               quadrotor_parameters=quadrotors_parameters[i],
                               enable_rotor_dynamics=enable_rotor_dynamics,
                               enable_rotor_drag=enable_rotor_drag,
                               enable_fuselage_drag=enable_fuselage_drag,
                               time_step=simulation_step,
                               )

    def init_obstacles(self,
                       obstacle_descriptions: List[str],
                       obstacle_poses: List[ArrayLike],
                       ):
        for i in range(len(obstacle_descriptions)):
            self.add_obstacle(obstacle_description=obstacle_descriptions[i],
                              obstacle_pose=obstacle_poses[i],
                              )

    def add_obstacle(self,
                     obstacle_description: str,
                     obstacle_pose: ArrayLike,
                     ):
        """Load an obstacle into the simulation"""
        pose_arr = np.array(obstacle_pose)
        pos = pose_arr[:3]
        quat = pose_arr[3:]
        self.obstacle_bullet_ids.append(p.loadURDF(fileName=obstacle_description,
                                                   basePosition=pos,
                                                   baseOrientation=quat,
                                                   physicsClientId=self._physics_client_id))

    def add_quadrotor(self,
                      quadrotor_description: str,
                      quadrotor_initial_pose: ArrayLike,
                      quadrotor_initial_twist: ArrayLike,
                      quadrotor_initial_rotor_speed: ArrayLike,
                      quadrotor_parameters: dict,
                      enable_rotor_dynamics: bool,
                      enable_rotor_drag: bool,
                      enable_fuselage_drag: bool,
                      time_step: float,
                      ):
        """Load a quadrotor into the simulation"""
        # Load the quadrotor into pybullet
        self.quadrotor_bullet_ids.append(p.loadURDF(quadrotor_description,
                                                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                                                    physicsClientId=self._physics_client_id))
        for i in range(-1, 4):
            p.changeDynamics(self.quadrotor_bullet_ids[-1], i, linearDamping=0, angularDamping=0, physicsClientId=self._physics_client_id)

        # Create the quadrotor physics object
        config = {
            'enable_rotor_dynamics': enable_rotor_dynamics,
            'enable_rotor_drag': enable_rotor_drag,
            'enable_fuselage_drag': enable_fuselage_drag,
            'TIME_STEP': time_step,
        }
        self.quadrotor_physics_objects.append(QuadrotorPhysics(params=quadrotor_parameters,
                                                               config=config))

        # Set the initial state of the quadrotor
        self.reset_quadrotor_state_rotor_speeds(self.num_quadrotos - 1,
                                                pose=quadrotor_initial_pose,
                                                twist=quadrotor_initial_twist,
                                                rotor_speed=quadrotor_initial_rotor_speed)

    def reset_quadrotor_state_rotor_speeds(self,
                                           quadrotor_index: int,
                                           pose: Union[ArrayLike, None] = None,
                                           twist: Union[ArrayLike, None] = None,
                                           rotor_speed: Union[ArrayLike, None] = None,
                                           ):
        if pose is not None:
            pose_arr = np.array(pose)
            pos = pose_arr[:3]
            quat = pose_arr[3:]
            p.resetBasePositionAndOrientation(self.quadrotor_bullet_ids[quadrotor_index],
                                              pos,
                                              quat,
                                              physicsClientId=self._physics_client_id)
            self.quadrotor_physics_objects[quadrotor_index].set_state(pos=pos, quat=quat)
        if twist is not None:
            twist_arr = np.array(twist)
            lin_vel = twist_arr[:3]
            ang_vel = twist_arr[3:]
            p.resetBaseVelocity(self.quadrotor_bullet_ids[quadrotor_index],
                                lin_vel,
                                ang_vel,
                                physicsClientId=self._physics_client_id)
            self.quadrotor_physics_objects[quadrotor_index].set_state(vel=lin_vel, ang_vel=ang_vel)
        if rotor_speed is not None:
            self.quadrotor_physics_objects[quadrotor_index].set_rotor_speeds(rotor_speed)

    def __del__(self):
        """
        Close PyBullet simulation.
        """
        p.disconnect(self._physics_client_id)
