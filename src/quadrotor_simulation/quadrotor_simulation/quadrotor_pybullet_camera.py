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

from scipy.spatial.transform import Rotation

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_FREQUENCY_IMG = 5  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorPybulletCamera(Node):

    def __init__(self):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_camera_node')

        # Declare the parameters
        self.declare_parameter('physics_server', 'GUI')  # GUI, DIRECT
        self.declare_parameter('quadrotor_description', 'cf2x')
        self.declare_parameter('obstacles_description', ['NONE'])
        self.declare_parameter('obstacles_poses', [0.0])
        self.declare_parameter('render_ground', True)
        self.declare_parameter('render_architecture', True)
        self.declare_parameter('image_width', 800)
        self.declare_parameter('image_height', 600)
        self.declare_parameter('image_publishing_frequency', DEFAULT_FREQUENCY_IMG)
        self.declare_parameter('state_topic', 'quadrotor_state')
        self.declare_parameter('image_topic', 'quadrotor_img')

        # Get the parameters
        self.physics_server = self.get_parameter_value('physics_server', 'str')
        self.quadrotor_description_file_name = self.get_parameter_value('quadrotor_description', 'str')
        self.obstacles_description_file_names = self.get_parameter_value('obstacles_description', 'list[str]')
        self.obstacles_poses = self.get_parameter_value('obstacles_poses', 'list[float]')
        self.render_ground = self.get_parameter_value('render_ground', 'bool')
        self.render_architecture = self.get_parameter_value('render_architecture', 'bool')
        self.image_width = self.get_parameter_value('image_width', 'int')
        self.image_height = self.get_parameter_value('image_height', 'int')
        self.image_publishing_frequency = self.get_parameter_value('image_publishing_frequency', 'int')
        self.state_topic = self.get_parameter_value('state_topic', 'str')
        self.image_topic = self.get_parameter_value('image_topic', 'str')

        # Subscribers and Publishers
        self.state_subscriber = self.create_subscription(msg_type=State,
                                                         topic=self.state_topic,
                                                         callback=self.receive_state_callback,
                                                         qos_profile=DEFAULT_QOS_PROFILE)
        self.image_publisher = self.create_publisher(msg_type=Image,
                                                     topic=self.image_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)

        # Control the frequencies of pbulishing
        self.image_publishing_period = 1.0 / self.image_publishing_frequency  # seconds

        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_pybullet()

        # Initialize the published and received data
        self.initialize_data()

        # initialize timers
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

        self.state = State()
        self.ros_img = Image()

    def receive_state_callback(self, msg):

        self.state = msg

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
        pose = self.state.state.pose
        quad_pos = [pose.position.x, pose.position.y, pose.position.z]
        quad_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        p.resetBasePositionAndOrientation(self.quadrotor_id, quad_pos, quad_quat)

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

    node = QuadrotorPybulletCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
