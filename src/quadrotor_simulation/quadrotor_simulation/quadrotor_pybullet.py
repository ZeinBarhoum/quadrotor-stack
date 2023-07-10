#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from ament_index_python.packages import get_package_share_directory
from quadrotor_interfaces.msg import RotorCommand, State

from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist

import pybullet as p
import pybullet_data
import xacro
import os

import numpy as np

import cv2

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

class QuadrotorPybullet(Node):
    def __init__(self):
        super().__init__('quadrotor_pybullet_node')

        self.start_time = self.get_clock().now()  # For logging purposes

        # Create a subscriber to receive the rotor commands
        self.subscription = self.create_subscription(
            RotorCommand,
            'quadrotor_rotor_speeds',
            self.receive_commands_callback,
            10  # Queue size
        )

        # Create a publisher to publish the quadrotor state
        self.publisher = self.create_publisher(
            State,
            'quadrotor_state',
            10  # Queue size
        )

        self.image_publisher = self.create_publisher(Image, '/quadrotor_img', 10)


        # Control the simulation frequency
        self.simulation_frequency = 240  # Hz
        self.timer_period = 1.0 / self.simulation_frequency  # seconds
        self.state_timer = self.create_timer(self.timer_period, self.simulation_step)

        self.image_publishing_frequency = 24  # Change this value as desired
        self.image_publishing_period = 1.0 / self.image_publishing_frequency  # seconds
        self.camera_timer = self.create_timer(self.image_publishing_period, self.publish_image)



        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_constants()
        self.initialize_pybullet()

        # initialize control commands
        self.rotor_speeds = np.array([self.HOVER_RPM] * 4)

        # Announce that the node is initialized
        self.get_logger().info('Simulator node initialized')

    def initialize_constants(self):
        self.G = 9.81  # m/s^2
        self.KF = 3.16e-10  # N/(rad/s)^2
        self.KM = 7.94e-12  # Nm/(rad/s)^2
        self.M = 0.027  # kg
        self.W = self.M*self.G  # N
        self.HOVER_RPM = np.sqrt(self.W/(4*self.KF))  # rpm

    def initialize_urdf(self):
        """Read the xacro file and convert it to a urdf file that can be read by pybullet."""
        description_folder = os.path.join(
            get_package_share_directory('quadrotor_description'), 'description')

        description_file = os.path.join(description_folder, 'cf2x.urdf.xacro')

        robot_description_content = xacro.process_file(
            description_file).toxml()

        new_file = os.path.join(description_folder, 'cf2x.urdf')

        with open(new_file, 'w+') as f:
            f.write(robot_description_content)

        self.urdf_file = new_file

    def initialize_pybullet(self):
        self.physicsClient = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -self.G)

        self.planeId = p.loadURDF("plane.urdf")

        self.quadrotor_id = p.loadURDF(self.urdf_file, [0, 0, 0.25])

        p.loadURDF("samurai.urdf")

    def receive_commands_callback(self, msg):
        # Process the received RotorCommand message here (e.g., extract rotor speeds)
        # Perform the simulation step and publish the updated state
        # self.get_logger().info(f'I recived the commands {msg}')
        self.rotor_speeds = np.array(msg.rotor_speeds)

    def get_F_T(self):
        w_rpm = self.rotor_speeds
        F = np.array(w_rpm**2)*self.KF
        T = np.array(w_rpm**2)*self.KM
        Tz = (-T[0] + T[1] - T[2] + T[3])
        return F, np.array([0, 0, Tz])

    def simulation_step(self):
        # Perform the simulation step here
        # Publish the updated state
        # self.get_logger().info(
            # f'Simulation step at {self.get_clock().now().seconds_nanoseconds()}')
        F, T = self.get_F_T()

        for i in range(4):
            p.applyExternalForce(self.quadrotor_id, i, forceObj=[
                                 0, 0, F[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)

        # applying Tz on the center of mass, the only one that depend on the drag and isn't simulated by the forces before
        p.applyExternalTorque(self.quadrotor_id, 4,
                              torqueObj=T, flags=p.LINK_FRAME)

        p.stepSimulation()

        self.publish_state()

    def publish_state(self):
        quad_pos, quad_quat = p.getBasePositionAndOrientation(
            self.quadrotor_id)
        self.quad_pos = quad_pos
        self.quad_quat = quad_quat

        quad_v, quad_w = p.getBaseVelocity(self.quadrotor_id)

        pose = Pose()
        pose.position = Point(x=quad_pos[0], y=quad_pos[1], z=quad_pos[2])
        pose.orientation = Quaternion(
            x=quad_quat[0], y=quad_quat[1], z=quad_quat[2], w=quad_quat[3])

        twist = Twist()
        twist.linear = Vector3(x=quad_v[0], y=quad_v[1], z=quad_v[2])
        twist.angular = Vector3(x=quad_w[0], y=quad_w[1], z=quad_w[2])

        msg = State()
        msg.header.stamp =Time(nanoseconds = self.get_clock().now().nanoseconds - self.start_time.nanoseconds).to_msg()
        msg.pose = pose
        msg.twist = twist

        #publish the message
        self.publisher.publish(msg)

    def publish_image(self):
        # Publish the image here
        # Capture the POV image
        bridge = CvBridge()
        # fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        # projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

        quad_pos, quad_quat = self.quad_pos, self.quad_quat
        rot_matrix = p.getMatrixFromQuaternion(quad_quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0) # z-axis
        init_up_vector = (0, 0, 1) # y-axis
        # # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix = p.computeViewMatrix(quad_pos, quad_pos + 0.1 * camera_vector, up_vector)

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(800) / 600, nearVal=0.1, farVal=1000.0
        )

        _, _, px, _, _ = p.getCameraImage(
            width=800,
            height=600,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            flags= p.ER_NO_SEGMENTATION_MASK,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert the image to ROS format
        image_rgb = np.array(px, dtype=np.uint8)
        image_rgb = np.reshape(image_rgb, (600, 800, 4))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        ros_image = bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")

        # Publish the image
        self.image_publisher.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)

    node = QuadrotorPybullet()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
