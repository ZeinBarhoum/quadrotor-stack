#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from quadrotor_interfaces.msg import RotorCommand, State
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist
from sensor_msgs.msg import Image

import pybullet as p
import pybullet_data

import cv2
from cv_bridge import CvBridge

import xacro
import os
import numpy as np
import yaml

from timeit import timeit

from typing import Union, List

from scipy.spatial.transform import Rotation

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorPybulletPhysics(Node):

    def __init__(self):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_physics_node')

        # Declare the parameters
        self.declare_parameters(namespace='', parameters=[('physics_server', 'DIRECT'),  # GUI, DIRECT
                                                          ('quadrotor_description', 'cf2x'),
                                                          ('obstacles_description', ['NONE']),
                                                          ('obstacles_poses', [0.0]),
                                                          ('render_ground', True),
                                                          ('simulation_step_frequency', DEFAULT_FREQUENCY),
                                                          ('state_topic', 'quadrotor_state'),
                                                          ('rotor_speeds_topic', 'quadrotor_rotor_speeds')])
        # Get the parameters
        self.physics_server = self.get_parameter('physics_server').get_parameter_value().string_value
        self.quadrotor_description_file_name = self.get_parameter('quadrotor_description').get_parameter_value().string_value
        self.obstacles_description_file_names = self.get_parameter('obstacles_description').get_parameter_value().string_array_value
        self.obstacles_poses = self.get_parameter('obstacles_poses').get_parameter_value().double_array_value
        self.render_ground = self.get_parameter('render_ground').get_parameter_value().bool_value
        self.simulation_step_frequency = self.get_parameter('simulation_step_frequency').get_parameter_value().integer_value
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self.rotor_speeds_topic = self.get_parameter('rotor_speeds_topic').get_parameter_value().string_value

        # # Declare the parameters
        # self.declare_parameter('physics_server', 'DIRECT')  # GUI, DIRECT
        # self.declare_parameter('quadrotor_description', 'cf2x')
        # self.declare_parameter('obstacles_description', ['NONE'])
        # self.declare_parameter('obstacles_poses', [0.0])
        # self.declare_parameter('render_ground', True)
        # self.declare_parameter('simulation_step_frequency', DEFAULT_FREQUENCY)
        # self.declare_parameter('state_topic', 'quadrotor_state')
        # self.declare_parameter('rotor_speeds_topic', 'quadrotor_rotor_speeds')

        # # Get the parameters
        # self.physics_server = self.get_parameter_value('physics_server', 'str')
        # self.quadrotor_description_file_name = self.get_parameter_value('quadrotor_description', 'str')
        # self.obstacles_description_file_names = self.get_parameter_value('obstacles_description', 'list[str]')
        # self.obstacles_poses = self.get_parameter_value('obstacles_poses', 'list[float]')
        # self.render_ground = self.get_parameter_value('render_ground', 'bool')
        # self.simulation_step_frequency = self.get_parameter_value('simulation_step_frequency', 'int')
        # self.state_topic = self.get_parameter_value('state_topic', 'str')
        # self.rotor_speeds_topic = self.get_parameter_value('rotor_speeds_topic', 'str')

        # Subscribers and Publishers
        self.rotor_speeds_subscriber = self.create_subscription(msg_type=RotorCommand,
                                                                topic=self.rotor_speeds_topic,
                                                                callback=self.receive_commands_callback,
                                                                qos_profile=DEFAULT_QOS_PROFILE)
        self.state_publisher = self.create_publisher(msg_type=State,
                                                     topic=self.state_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)

        # Control the frequencies of simulation and pbulishing
        self.simulation_step_period = 1.0 / self.simulation_step_frequency  # seconds

        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_constants()
        self.initialize_pybullet()

        # Initialize the published and received data
        self.initialize_data()

        # initialize timers
        self.simulation_step_timer = self.create_timer(self.simulation_step_period, self.simulation_step_callback)

        # Announce that the node is initialized
        self.start_time = self.get_clock().now()  # For logging purposes
        self.get_logger().info(f'QuadrotorPybulletPhysics node initialized at {self.start_time.seconds_nanoseconds()}')

    def initialize_constants(self):
        """ Initializes the physical constants of the quadrotor model by loading them from a configuration file.
        The configuration file is expected to be located in the 'quadrotor_description/config' folder and named '<name>_params.yaml' 
        where <name> is the name of the quadrotor description file.
        The loaded parameters are stored as instance variables for later use in the simulation.
        Parameters:
            None

        Returns:
            None
        """
        config_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'config')
        config_file = os.path.join(config_folder, self.quadrotor_description_file_name+'_params.yaml')
        with open(config_file, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self.get_logger().error(
                    f"Cofiguration File {config_file} Couldn't Be Loaded, Raised Error {exc}")
                parameters = dict()
        # self.get_logger().info(f'{parameters=}')
        CF2X_PARAMS = parameters['CF2X_PARAMS']
        self.G = 9.81  # m/s^2
        self.KF = CF2X_PARAMS['KF']  # N/(rad/s)^2
        self.KM = CF2X_PARAMS['KM']  # Nm/(rad/s)^2
        self.M = CF2X_PARAMS['M']  # kg
        self.W = self.M*self.G  # N
        self.HOVER_RPM = np.sqrt(self.W/(4*self.KF))  # rad/s

    def initialize_urdf(self):
        """
        Initializes the quadrotor and obstacle URDF files for PyBullet simulation.

        This method uses Xacro to convert the quadrotor's Xacro file to a URDF file and saves it for PyBullet to read.
        It also retrieves the obstacle URDF files and saves them for PyBullet to read.

        Args:
            None

        Returns:
            None
        """
        quadrotor_description_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'description')
        quadrotor_description_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf.xacro')
        quadrotor_description_content = xacro.process_file(quadrotor_description_file).toxml()
        new_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf')
        with open(new_file, 'w+') as f:
            f.write(quadrotor_description_content)
        self.quadrotor_urdf_file = new_file

        # Retreive the obstacle urdf file and save it for pybullet to read
        obstacles_description_folder = os.path.join(get_package_share_directory('quadrotor_simulation'), 'world')
        self.obstacle_urdf_files = []
        for name in self.obstacles_description_file_names:
            if (name == 'NONE'):
                break
            self.obstacle_description_file_name = name
            obstacle_description_file = os.path.join(obstacles_description_folder, self.obstacle_description_file_name+'.urdf.xacro')
            obstacle_description_content = xacro.process_file(obstacle_description_file).toxml()
            new_file = os.path.join(obstacles_description_folder, name + '.urdf')
            with open(new_file, 'w+') as f:
                f.write(obstacle_description_content)
            self.obstacle_urdf_files.append(new_file)

    def initialize_pybullet(self):
        """
        Initializes the PyBullet physics engine and loads the necessary URDF files for the quadrotor simulation.
        If the physics server is set to 'DIRECT', the simulation runs without a GUI, otherwise a GUI is displayed.
        The simulation time step period, gravity, ground and obstacles are also set up.

        Args:
            None

        Returns:
            None
        """
        if (self.physics_server == 'DIRECT'):
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.simulation_step_period)
        p.setGravity(0, 0, -self.G)
        if (self.render_ground):
            self.planeId = p.loadURDF("plane.urdf")
        self.obstacleIds = []
        for (i, obstacle_urdf_file) in enumerate(self.obstacle_urdf_files):
            self.obstacleIds.append(p.loadURDF(obstacle_urdf_file, self.obstacles_poses[i*7: i*7+3], self.obstacles_poses[i*7+3: i*7+7], useFixedBase=1))
        self.quadrotor_id = p.loadURDF(self.quadrotor_urdf_file, [0, 0, 0.25])
        # Disable default damping of pybullet!
        p.changeDynamics(self.quadrotor_id, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 2, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 3, linearDamping=0, angularDamping=0)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    def initialize_data(self):
        """
        Initializes the data required for the quadrotor simulation.

        Sets the initial rotor speeds to the hover RPM, initializes the state of the quadrotor, and creates an empty ROS image.

        Args:
            None

        Returns:
            None
        """

        self.rotor_speeds = np.array([self.HOVER_RPM] * 4)
        self.state = State()
        self.ros_img = Image()

    def receive_commands_callback(self, msg):
        """
        Callback function to receive commands from a ROS topic.

        Args:
            msg: A ROS message containing the rotor speeds.

        Returns:
            None
        """
        self.rotor_speeds = np.array(msg.rotor_speeds)

    def get_F_T(self):
        """
        Computes the total force and torque generated by the quadrotor's rotors, based on their current speeds.

        Returns:
        - F: a 1D numpy array of shape (4,), representing the total force generated by the rotors, in Newtons.
        - T: a 1D numpy array of shape (3,), representing the total torque generated by the rotors (along z-axis), in Newton-meters.
            The first two components of T are always zero, since the torques aroung these axes are simulated by the forces applied to the rotors.
            The third component of T represents the net torque (due to aerodynamic drag) around the z axis, which controls the quadrotor's yaw rotation.
        """
        w_rpm = self.rotor_speeds
        F = np.array(w_rpm**2)*self.KF
        T = np.array(w_rpm**2)*self.KM
        Tz = (-T[0] + T[1] - T[2] + T[3])
        return F, np.array([0, 0, Tz])

    def simulation_step_callback(self):
        """
        Callback function that is called at each simulation step. Calculates the forces and torques to be applied to the quadrotor,
        applies them to the simulation, and updates the quadrotor's state. The state is then stored in the `self.state` attribute.

        Returns:
            None
        """
        F, T = self.get_F_T()  # calculate the forces and torques
        for i in range(4):  # for each rotor
            p.applyExternalForce(self.quadrotor_id, i, forceObj=[0, 0, F[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)

        # applying Tz on the center of mass, the only one that depend on the drag and isn't simulated by the forces before
        p.applyExternalTorque(self.quadrotor_id, 4, torqueObj=T, flags=p.LINK_FRAME)

        p.stepSimulation()

        quad_pos, quad_quat = p.getBasePositionAndOrientation(self.quadrotor_id)

        self.quad_pos = quad_pos
        self.quad_quat = quad_quat

        quad_v, quad_w = p.getBaseVelocity(self.quadrotor_id)
        quad_w = Rotation.from_quat(quad_quat).inv().apply(quad_w)

        pose = Pose()
        pose.position = Point(x=quad_pos[0], y=quad_pos[1], z=quad_pos[2])
        pose.orientation = Quaternion(x=quad_quat[0], y=quad_quat[1], z=quad_quat[2], w=quad_quat[3])

        twist = Twist()
        twist.linear = Vector3(x=quad_v[0], y=quad_v[1], z=quad_v[2])
        twist.angular = Vector3(x=quad_w[0], y=quad_w[1], z=quad_w[2])

        self.state = State()
        self.state.header.stamp = self.get_clock().now().to_msg()
        self.state.state.pose = pose
        self.state.state.twist = twist
        self.state.quadrotor_id = self.quadrotor_id

        self.state_publisher.publish(self.state)


def main(args=None):
    """
    Initializes the ROS 2 node for the quadrotor simulation using PyBullet physics engine.
    Args:
        args: List of strings representing command line arguments.
    """
    rclpy.init(args=args)

    node = QuadrotorPybulletPhysics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
