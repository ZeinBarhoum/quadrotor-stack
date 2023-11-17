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

import threading

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

    def __init__(self, suffix='', **kwargs):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_camera_node'+suffix, **kwargs)

        # Declare parameters
        self.declare_parameters(namespace='', parameters=[('physics_server', 'GUI'),
                                                          ('quadrotor_description', 'cf2x'),
                                                          ('quadrotor_scale', 1.0),
                                                          ('obstacles_description', ['NONE']),
                                                          ('obstacles_poses', [0.0]),
                                                          ('render_ground', True),
                                                          ('render_architecture', True),
                                                          ('image_publishing_frequency', DEFAULT_FREQUENCY_IMG),
                                                          ('state_topic', 'quadrotor_state'+suffix),
                                                          ('image_topic', 'quadrotor_img'+suffix),
                                                          ('threading', False),
                                                          ('image_width', 800),
                                                          ('image_height', 600),
                                                          ('camera_position', [0.0, 0.0, 0.0]),
                                                          ('camera_main_axis', [1.0, 0.0, 0.0]),
                                                          ('camera_up_axis', [0.0, 0.0, 1.0]),
                                                          ('camera_focus_distance', 1.0),
                                                          ('camera_fov', 60.0),
                                                          ('camera_near_plane', 0.01),
                                                          ('camera_far_plane', 100.0),
                                                          ('sequential_mode', False),
                                                          ])

        # Get the parameters
        self.physics_server = self.get_parameter('physics_server').get_parameter_value().string_value
        self.quadrotor_description_file_name = self.get_parameter('quadrotor_description').get_parameter_value().string_value
        self.quadrotor_scale = self.get_parameter('quadrotor_scale').get_parameter_value().double_value
        self.obstacles_description_file_names = self.get_parameter('obstacles_description').get_parameter_value().string_array_value
        self.obstacles_poses = self.get_parameter('obstacles_poses').get_parameter_value().double_array_value
        self.render_ground = self.get_parameter('render_ground').get_parameter_value().bool_value
        self.render_architecture = self.get_parameter('render_architecture').get_parameter_value().bool_value
        self.image_publishing_frequency = self.get_parameter('image_publishing_frequency').get_parameter_value().integer_value
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.threading = self.get_parameter('threading').get_parameter_value().bool_value
        self.sequential_mode = self.get_parameter('sequential_mode').get_parameter_value().bool_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value

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
        if not self.sequential_mode:
            self.image_publishing_timer = self.create_timer(self.image_publishing_period, self.publish_image_callback)

        # Announce that the node is initialized
        self.start_time = self.get_clock().now()  # For logging purposes
        self.get_logger().info(f'QuadrotorPybulletCamera node initialized at {self.start_time.seconds_nanoseconds()}')

    def initialize_urdf(self):
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

        if (self.physics_server == 'DIRECT'):
            self.physicsClient = p.connect(p.DIRECT)
        elif (self.physics_server == 'SHARED_MEMORY'):
            self.physicsClient = p.connect(p.SHARED_MEMORY)
        else:
            self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if (self.render_ground):
            self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        if (self.render_architecture):
            p.loadURDF("samurai.urdf", physicsClientId=self.physicsClient)
        self.obstacleIds = []
        for (i, obstacle_urdf_file) in enumerate(self.obstacle_urdf_files):
            self.obstacleIds.append(p.loadURDF(obstacle_urdf_file, self.obstacles_poses[i*7: i*7+3], self.obstacles_poses[i*7+3: i*7+7], useFixedBase=1))
        self.quadrotor_id = p.loadURDF(self.quadrotor_urdf_file, [0, 0, 0.25], globalScaling=self.quadrotor_scale, physicsClientId=self.physicsClient)
        # Configure the debug visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)

    def initialize_data(self):
        self.state = State()
        self.ros_image = Image()
        self.bridge = CvBridge()

    def receive_state_callback(self, msg):
        self.state = msg
        if self.sequential_mode:
            self.publish_image_callback()

    def publish_image_callback(self):
        if (self.threading):
            threading.Thread(target=self.publish_image_callback_thread).start()
        else:
            self.publish_image_callback_thread()

    def publish_image_callback_thread(self):
        self.image_publishing_frequency = self.get_parameter('image_publishing_frequency').get_parameter_value().integer_value
        if not self.sequential_mode:
            self.image_publishing_timer.timer_period_ns = int(1e9 / self.image_publishing_frequency)

        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.camera_position = self.get_parameter('camera_position').get_parameter_value().double_array_value
        self.camera_main_axis = self.get_parameter('camera_main_axis').get_parameter_value().double_array_value
        self.camera_up_axis = self.get_parameter('camera_up_axis').get_parameter_value().double_array_value
        self.camera_focus_distance = self.get_parameter('camera_focus_distance').get_parameter_value().double_value
        self.camera_fov = self.get_parameter('camera_fov').get_parameter_value().double_value
        self.camera_near_plane = self.get_parameter('camera_near_plane').get_parameter_value().double_value
        self.camera_far_plane = self.get_parameter('camera_far_plane').get_parameter_value().double_value

        pose = self.state.state.pose
        quad_pos = [pose.position.x, pose.position.y, pose.position.z]
        quad_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        p.resetBasePositionAndOrientation(self.quadrotor_id, quad_pos, quad_quat, physicsClientId=self.physicsClient)

        rot_matrix = p.getMatrixFromQuaternion(quad_quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        init_camera_vector = np.array(self.camera_main_axis)/np.linalg.norm(self.camera_main_axis)
        init_up_vector = np.array(self.camera_up_axis)/np.linalg.norm(self.camera_up_axis)
        # # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        camera_pos = rot_matrix.dot(np.array(self.camera_position)) + np.array(quad_pos)
        camera_focus_target = camera_pos + camera_vector*self.camera_focus_distance

        view_matrix = p.computeViewMatrix(
            camera_pos, camera_focus_target, up_vector)

        projection_matrix = p.computeProjectionMatrixFOV(fov=self.camera_fov, aspect=float(
            self.image_width) / self.image_height, nearVal=self.camera_near_plane, farVal=self.camera_far_plane)

        _, _, px, _, _ = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physicsClient
        )

        # Convert the image to ROS format
        image_rgb = np.array(px, dtype=np.uint8)
        image_rgb = np.reshape(image_rgb, (self.image_height, self.image_width, 4))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        self.ros_image = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
        # Publish the image
        self.image_publisher.publish(self.ros_image)


def main(args=None):

    rclpy.init(args=args)
    node = QuadrotorPybulletCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
