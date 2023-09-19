#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time

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

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_FREQUENCY_IMG = 5  # Hz
DEFAULT_QOS_PROFILE = 10
DEFAULT_VIEW_DISTANCE = 5.0
DEFAULT_VIEW_YAW = 0.0
DEFAULT_VIEW_PITCH = 0.0


class QuadrotorPybullet(Node):
    """
    A ROS2 node that simulates a quadrotor using PyBullet physics engine.

    This node subscribes to a topic for rotor speeds and publishes the state of the quadrotor and an image of the
    simulation. It also renders the ground and obstacles in the simulation.

    Parameters:
        physics_server (str): The type of PyBullet physics server to use. Can be 'GUI' or 'DIRECT'.
        quadrotor_description (str): The name of the URDF file that describes the quadrotor.
        obstacles_description (List[str]): The names of the URDF files that describe the obstacles in the simulation.
        render_ground (bool): Whether to render the ground in the simulation.
        render_architecture (bool): Whether to render the environment in the simulation.
        publish_image (bool): Whether to publish an image of the simulation.
        image_width (int): The width of the image to publish.
        image_height (int): The height of the image to publish.
        state_publishing_frequency (int): The frequency at which to publish the state of the quadrotor.
        image_publishing_frequency (int): The frequency at which to publish the image of the simulation.
        simulation_step_frequency (int): The frequency at which to step the simulation.
        state_topic (str): The name of the topic to publish the state of the quadrotor.
        image_topic (str): The name of the topic to publish the image of the simulation.
        rotor_speeds_topic (str): The name of the topic to subscribe to for rotor speeds.
        view_distance (float): The distance of the camera from the quadrotor in the view.
        view_yaw (float): The yaw angle of the camera in the view.
        view_pitch (float): The pitch angle of the camera in the view.

    Subscribers:
        rotor_speeds_subscriber (RotorCommand): Subscribes to the topic for rotor speeds.

    Publishers:
        state_publisher (State): Publishes the state of the quadrotor.
        image_publisher (Image): Publishes an image of the simulation.

    Timers:
        simulation_step_timer: Steps the simulation at the specified frequency.
        state_publishing_timer: Publishes the state of the quadrotor at the specified frequency.
        image_publishing_timer: Publishes an image of the simulation at the specified frequency.

    """

    def __init__(self):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_node')

        # Declare the parameters
        self.declare_parameter('physics_server', 'GUI')  # GUI, DIRECT
        self.declare_parameter('quadrotor_description', 'cf2x')
        self.declare_parameter('obstacles_description', ['NONE'])
        self.declare_parameter('obstacles_poses', [0.0])
        self.declare_parameter('render_ground', True)
        self.declare_parameter('render_architecture', True)
        self.declare_parameter('publish_image', True)
        self.declare_parameter('image_width', 800)
        self.declare_parameter('image_height', 600)
        self.declare_parameter('state_publishing_frequency', DEFAULT_FREQUENCY)
        self.declare_parameter('image_publishing_frequency', DEFAULT_FREQUENCY_IMG)
        self.declare_parameter('simulation_step_frequency', DEFAULT_FREQUENCY)
        self.declare_parameter('state_topic', 'quadrotor_state')
        self.declare_parameter('image_topic', 'quadrotor_img')
        self.declare_parameter('rotor_speeds_topic', 'quadrotor_rotor_speeds')
        self.declare_parameter('view_distance', DEFAULT_VIEW_DISTANCE)
        self.declare_parameter('view_yaw', DEFAULT_VIEW_YAW)
        self.declare_parameter('view_pitch', DEFAULT_VIEW_PITCH)

        # Get the parameters
        self.physics_server = self.get_parameter_value('physics_server', 'str')
        self.quadrotor_description_file_name = self.get_parameter_value('quadrotor_description', 'str')
        self.obstacles_description_file_names = self.get_parameter_value('obstacles_description', 'list[str]')
        self.obstacles_poses = self.get_parameter_value('obstacles_poses', 'list[float]')
        self.render_ground = self.get_parameter_value('render_ground', 'bool')
        self.render_architecture = self.get_parameter_value('render_architecture', 'bool')
        self.publish_image = self.get_parameter_value('publish_image', 'bool')
        self.image_width = self.get_parameter_value('image_width', 'int')
        self.image_height = self.get_parameter_value('image_height', 'int')
        self.state_publishing_frequency = self.get_parameter_value('state_publishing_frequency', 'int')
        self.image_publishing_frequency = self.get_parameter_value('image_publishing_frequency', 'int')
        self.simulation_step_frequency = self.get_parameter_value('simulation_step_frequency', 'int')
        self.state_topic = self.get_parameter_value('state_topic', 'str')
        self.image_topic = self.get_parameter_value('image_topic', 'str')
        self.rotor_speeds_topic = self.get_parameter_value('rotor_speeds_topic', 'str')
        self.view_distance = self.get_parameter_value('view_distance', 'float')
        self.view_yaw = self.get_parameter_value('view_yaw', 'float')
        self.view_pitch = self.get_parameter_value('view_pitch', 'float')

        # Subscribers and Publishers
        self.rotor_speeds_subscriber = self.create_subscription(msg_type=RotorCommand,
                                                                topic=self.rotor_speeds_topic,
                                                                callback=self.receive_commands_callback,
                                                                qos_profile=DEFAULT_QOS_PROFILE)
        self.state_publisher = self.create_publisher(msg_type=State,
                                                     topic=self.state_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)
        self.image_publisher = self.create_publisher(msg_type=Image,
                                                     topic=self.image_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)

        # Control the frequencies of simulation and pbulishing
        self.simulation_step_period = 1.0 / self.simulation_step_frequency  # seconds
        self.state_publishing_period = 1.0 / self.state_publishing_frequency  # seconds
        self.image_publishing_period = 1.0 / self.image_publishing_frequency  # seconds

        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_constants()
        self.initialize_pybullet()

        # Initialize the published and received data
        self.initialize_data()

        # initialize timers
        self.simulation_step_timer = self.create_timer(self.simulation_step_period, self.simulation_step_callback)
        self.state_publishing_timer = self.create_timer(self.state_publishing_period, self.publish_state_callback)
        if self.publish_image:
            self.image_publishing_timer = self.create_timer(self.image_publishing_period, self.publish_image_callback)

        # Announce that the node is initialized
        self.start_time = self.get_clock().now()  # For logging purposes
        self.get_logger().info(f'QuadrotorPybullet node initialized at {self.start_time.seconds_nanoseconds()}')

    def get_parameter_value(self, parameter_name: str, parameter_type: str) -> Union[bool, int, float, str, List[str]]:
        """
        Get the value of a parameter with the given name and type.

        Args:
            parameter_name: The name of the parameter to retrieve.
            parameter_type: The type of the parameter to retrieve. Supported types are 'bool', 'int', 'float', 'str',
                'list[float]' and 'list[str]'.

        Returns:
            The value of the parameter, cast to the specified type.

        Raises:
            ValueError: If the specified parameter type is not supported.
        """

        parameter = self.get_parameter(parameter_name)
        parameter_value = parameter.get_parameter_value()

        if parameter_type == 'bool':
            return parameter_value.bool_value
        elif parameter_type == 'int':
            return parameter_value.integer_value
        elif parameter_type == 'float':
            return parameter_value.double_value
        elif parameter_type == 'str':
            return parameter_value.string_value
        elif parameter_type == 'list[str]':
            return parameter_value.string_array_value
        elif parameter_type == 'list[float]':
            return parameter_value.double_array_value
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type}")

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
        self.HOVER_RPM = np.sqrt(self.W/(4*self.KF))  # rpm

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
        if (self.render_architecture):
            p.loadURDF("samurai.urdf")
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

        p.resetDebugVisualizerCamera(cameraDistance=self.view_distance, cameraYaw=self.view_yaw,
                                     cameraPitch=self.view_pitch, cameraTargetPosition=quad_pos)  # fix camera onto model

        self.quad_pos = quad_pos
        self.quad_quat = quad_quat

        quad_v, quad_w = p.getBaseVelocity(self.quadrotor_id)

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

    def publish_state_callback(self):
        """
        Publishes the current state of the quadrotor to the state publisher.

        The state publisher is a ROS publisher that broadcasts the quadrotor's state
        to other nodes in the ROS network. This method retrieves the current state
        of the quadrotor and publishes it to the state publisher.

        Args:
            None

        Returns:
            None
        """
        self.state_publisher.publish(self.state)

    def publish_image_callback(self):
        """
        Publishes an image from the quadrotor's camera to a ROS topic.

        The image is captured using PyBullet's getCameraImage function, which
        simulates the quadrotor's camera. The image is then converted to ROS format
        and published to the image_publisher topic.

        Returns:
            None
        """
        bridge = CvBridge()
        quad_pos, quad_quat = self.quad_pos, self.quad_quat
        rot_matrix = p.getMatrixFromQuaternion(quad_quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0)  # z-axis
        init_up_vector = (0, 0, 1)  # y-axis
        # # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix = p.computeViewMatrix(
            quad_pos, quad_pos + 0.1 * camera_vector, up_vector)

        projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(self.image_width) / self.image_height, nearVal=0.1, farVal=1000.0)

        _, _, px, _, _ = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert the image to ROS format
        image_rgb = np.array(px, dtype=np.uint8)
        image_rgb = np.reshape(image_rgb, (self.image_height, self.image_width, 4))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        ros_image = bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")

        # Publish the image
        self.image_publisher.publish(ros_image)


def main(args=None):
    """
    Initializes the ROS 2 node for the quadrotor simulation using PyBullet physics engine.
    Args:
        args: List of strings representing command line arguments.
    """
    rclpy.init(args=args)

    node = QuadrotorPybullet()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
