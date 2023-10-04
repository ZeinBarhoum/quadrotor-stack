#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from ament_index_python.packages import get_package_share_directory

from quadrotor_interfaces.msg import RotorCommand, State, StateData, ModelErrors
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image

import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation

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


class QuadrotorPybulletDataset(Node):
    """
    A ROS2 node that test the accuracy of models using datasets.
    Unlike the normal pybullet node that subsribes to the rotor_speeds topi and publishes to the state topic, This node subsribes to the rotor_speeds and state topics and publishes to the model_error topic. 
    At the same time, this node visualizes the quadrotor actual state and the model "predicted" states according to the model.
    """

    def __init__(self):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_node')

        # Declare the parameters
        self.declare_parameters(parameters=[('physics_server', 'GUI'),  # GUI, DIRECT
                                            ('quadrotor_description', 'cf2x'),
                                            ('quadrotor_ghost_description', 'cf2x'),
                                            ('obstacles_description', ['NONE']),
                                            ('obstacles_poses', [0.0]),
                                            ('render_ground', True),
                                            ('render_architecture', True),
                                            ('publish_image', False),
                                            ('image_width', 800),
                                            ('image_height', 600),
                                            ('image_publishing_frequency', DEFAULT_FREQUENCY_IMG),
                                            ('simulation_step_frequency', DEFAULT_FREQUENCY),
                                            ('state_publishing_frequency', DEFAULT_FREQUENCY),
                                            ('state_topic', 'quadrotor_state'),
                                            ('ff_state_topic', 'quadrotor_ff_state'),
                                            ('image_topic', 'quadrotor_img'),
                                            ('model_error_topic', 'quadrotor_model_error'),
                                            ('rotor_speeds_topic', 'quadrotor_rotor_speeds'),
                                            ('view_follow', True),
                                            ('view_distance', DEFAULT_VIEW_DISTANCE),
                                            ('view_yaw', DEFAULT_VIEW_YAW),
                                            ('view_pitch', DEFAULT_VIEW_PITCH),
                                            ('use_ff_state', False),
                                            ],
                                namespace='',
                                )

        # Get the parameters
        self.physics_server = self.get_parameter_value('physics_server', 'str')
        self.quadrotor_description_file_name = self.get_parameter_value('quadrotor_description', 'str')
        self.quadrotor_ghost_description_file_name = self.get_parameter_value('quadrotor_ghost_description', 'str')
        self.obstacles_description_file_names = self.get_parameter_value('obstacles_description', 'list[str]')
        self.obstacles_poses = self.get_parameter_value('obstacles_poses', 'list[float]')
        self.render_ground = self.get_parameter_value('render_ground', 'bool')
        self.render_architecture = self.get_parameter_value('render_architecture', 'bool')
        self.publish_image = self.get_parameter_value('publish_image', 'bool')
        self.image_width = self.get_parameter_value('image_width', 'int')
        self.image_height = self.get_parameter_value('image_height', 'int')
        self.image_publishing_frequency = self.get_parameter_value('image_publishing_frequency', 'int')
        self.simulation_step_frequency = self.get_parameter_value('simulation_step_frequency', 'int')
        self.state_publishing_frequency = self.get_parameter_value('state_publishing_frequency', 'int')
        self.ff_state_topic = self.get_parameter_value('ff_state_topic', 'str')
        self.state_topic = self.get_parameter_value('state_topic', 'str')
        self.image_topic = self.get_parameter_value('image_topic', 'str')
        self.model_error_topic = self.get_parameter_value('model_error_topic', 'str')
        self.rotor_speeds_topic = self.get_parameter_value('rotor_speeds_topic', 'str')
        self.view_follow = self.get_parameter_value('view_follow', 'bool')
        self.view_distance = self.get_parameter_value('view_distance', 'float')
        self.view_yaw = self.get_parameter_value('view_yaw', 'float')
        self.view_pitch = self.get_parameter_value('view_pitch', 'float')
        self.use_ff_state = self.get_parameter_value('use_ff_state', 'bool')

        # Subscribers and Publishers
        self.rotor_speeds_subscriber = self.create_subscription(msg_type=RotorCommand,
                                                                topic=self.rotor_speeds_topic,
                                                                callback=self.receive_commands_callback,
                                                                qos_profile=DEFAULT_QOS_PROFILE)
        self.ff_state_subscriber = self.create_subscription(msg_type=State,
                                                            topic=self.ff_state_topic,
                                                            callback=self.receive_ff_state_callback,
                                                            qos_profile=DEFAULT_QOS_PROFILE)
        self.state_publisher = self.create_publisher(msg_type=State,
                                                     topic=self.state_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)
        self.image_publisher = self.create_publisher(msg_type=Image,
                                                     topic=self.image_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)
        self.model_error_publisher = self.create_publisher(msg_type=ModelErrors,
                                                           topic=self.model_error_topic,
                                                           qos_profile=DEFAULT_QOS_PROFILE)

        # Control the frequencies of simulation and pbulishing
        self.simulation_step_period = 1.0 / self.simulation_step_frequency  # seconds
        self.image_publishing_period = 1.0 / self.image_publishing_frequency  # seconds
        self.state_publishing_period = 1.0 / self.state_publishing_frequency  # seconds

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
        self.get_logger().info(f'QuadrotorPybulletDataset node initialized at {self.start_time.seconds_nanoseconds()}')

    def get_parameter_value(self, parameter_name: str, parameter_type: str) -> Union[bool, int, float, str, List[str]]:

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
        self.G = 9.81  # m/s^2
        self.KF = parameters['KF']  # N/(rpm)^2
        self.KM = parameters['KM']  # Nm/(rpm)^2
        self.M = parameters['Inertia']['M']  # kg
        self.J = np.diag([parameters['Inertia']['IXX'], parameters['Inertia']['IYY'], parameters['Inertia']['IZZ']])  # kg.m^2
        self.W = self.M*self.G  # N
        self.Config = parameters['Config']
        self.Rotor_Dirs = parameters['Rotor_Dirs']
        # self.HOVER_RPM = np.sqrt(self.W/(4*self.KF))  # rpm
        # self.get_logger().info(f"HOVER RPM IS : {self.HOVER_RPM} rpm")

    def initialize_urdf(self):
        quadrotor_description_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'description')
        quadrotor_description_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf.xacro')
        quadrotor_description_content = xacro.process_file(quadrotor_description_file).toxml()
        new_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf')
        with open(new_file, 'w+') as f:
            f.write(quadrotor_description_content)
        self.quadrotor_urdf_file = new_file

        quadrotor_ghost_description_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'description')
        quadrotor_ghost_description_file = os.path.join(quadrotor_description_folder, self.quadrotor_ghost_description_file_name+'.urdf.xacro')
        quadrotor_ghost_description_content = xacro.process_file(quadrotor_ghost_description_file).toxml()
        new_file = os.path.join(quadrotor_description_folder, self.quadrotor_ghost_description_file_name+'.urdf')
        with open(new_file, 'w+') as f:
            f.write(quadrotor_ghost_description_content)
        self.quadrotor_ghost_urdf_file = new_file

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
        self.quadrotor_id = p.loadURDF(self.quadrotor_urdf_file, [0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)
        # self.quadrotor_ghost_id = p.loadURDF(self.quadrotor_ghost_urdf_file, [0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)
        # self.quadrotor_ghost_id2 = p.loadURDF(self.quadrotor_ghost_urdf_file, [0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)

        # Disable default damping of pybullet!
        p.changeDynamics(self.quadrotor_id, -1, linearDamping=0, angularDamping=0, lateralFriction=0)
        p.changeDynamics(self.quadrotor_id, 0, linearDamping=0, angularDamping=0, lateralFriction=0)
        p.changeDynamics(self.quadrotor_id, 1, linearDamping=0, angularDamping=0, lateralFriction=0)
        p.changeDynamics(self.quadrotor_id, 2, linearDamping=0, angularDamping=0, lateralFriction=0)
        p.changeDynamics(self.quadrotor_id, 3, linearDamping=0, angularDamping=0, lateralFriction=0)

    def initialize_data(self):
        # received data
        self.rotor_speeds = np.array([0] * 4)
        self.ff_pos, self.ff_quat = np.array([0, 0, 0]), np.array([0, 0, 0, 1])
        self.ff_vel, self.ff_anvel = np.array([0, 0, 0]), np.array([0, 0, 0])
        self.ff_accel, self.ff_anaccel = np.array([0, 0, 0]), np.array([0, 0, 0])

        # intermediate data
        self.quad_pos, self.quad_quat = np.array([0, 0, 0]), np.array([0, 0, 0, 1])
        self.quad_vel, self.quad_avel = np.array([0, 0, 0]), np.array([0, 0, 0])
        self.quad_accel, self.quad_anaccel = np.array([0, 0, 0]), np.array([0, 0, 0])

        # published data
        self.state = State()
        self.ros_img = Image()
        self.model_error = ModelErrors()

    def receive_commands_callback(self, msg):
        self.rotor_speeds = np.array(msg.rotor_speeds)

    def receive_ff_state_callback(self, msg: State):
        self.ff_pos = np.array([msg.state.pose.position.x, msg.state.pose.position.y, msg.state.pose.position.z])
        self.ff_quat = np.array([msg.state.pose.orientation.x, msg.state.pose.orientation.y, msg.state.pose.orientation.z, msg.state.pose.orientation.w])
        self.ff_vel = np.array([msg.state.twist.linear.x, msg.state.twist.linear.y, msg.state.twist.linear.z])
        self.ff_anvel = np.array([msg.state.twist.angular.x, msg.state.twist.angular.y, msg.state.twist.angular.z])
        self.ff_accel = np.array([msg.state.accel.linear.x, msg.state.accel.linear.y, msg.state.accel.linear.z])
        self.ff_anaccel = np.array([msg.state.accel.angular.x, msg.state.accel.angular.y, msg.state.accel.angular.z])

    def get_rotor_thrust_quadratic(self, rotor_speed):
        # following the quadratic model, rotor_speed in rpm
        return self.KF * (rotor_speed**2)

    def get_rotor_torque_quadratic(self, rotor_speed):
        # following the quadratic model, rotor_speed in rpm
        return self.KM * (rotor_speed**2)

    def apply_forces_torques(self):
        rotor_speeds = self.rotor_speeds
        rotor_thrusts = [self.get_rotor_thrust_quadratic(speed) for speed in rotor_speeds]
        rotor_torques = [self.get_rotor_torque_quadratic(speed) for speed in rotor_speeds]
        torque_z = -(self.Rotor_Dirs[0]*rotor_torques[0] + self.Rotor_Dirs[1]*rotor_torques[1] +
                     self.Rotor_Dirs[2]*rotor_torques[2] + self.Rotor_Dirs[3]*rotor_torques[3])

        for i in range(4):
            p.applyExternalForce(self.quadrotor_id, i, forceObj=[0, 0, rotor_thrusts[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)
        # applying Tz on the center of mass, the only one that depend on the drag and isn't simulated by the forces before
        p.applyExternalTorque(self.quadrotor_id, -1, torqueObj=[0, 0, torque_z], flags=p.LINK_FRAME)

    def apply_ff_state(self):
        pos, quat = self.ff_pos, self.ff_quat
        vel, anvel_B = self.ff_vel, self.ff_anvel
        anvel_W = Rotation.from_quat(quat).apply(anvel_B)
        # for the actual quadrotor
        p.resetBasePositionAndOrientation(self.quadrotor_id, pos, quat)
        p.resetBaseVelocity(self.quadrotor_id, vel, anvel_W)

    def apply_simulation_step(self):
        pos, quat = p.getBasePositionAndOrientation(self.quadrotor_id)
        pos, quat = np.array(pos), np.array(quat)
        vel0, avel0_W = p.getBaseVelocity(self.quadrotor_id)
        vel0, avel0_B = np.array(vel0), Rotation.from_quat(quat).inv().apply(np.array(avel0_W))

        p.stepSimulation()

        pos, quat = p.getBasePositionAndOrientation(self.quadrotor_id)
        pos, quat = np.array(pos), np.array(quat)
        vel, avel_W = p.getBaseVelocity(self.quadrotor_id)
        vel, avel_B = np.array(vel), Rotation.from_quat(quat).inv().apply(np.array(avel_W))
        accel, anaccel = (vel-vel0)/self.simulation_step_period, (avel_B-avel0_B)/self.simulation_step_period

        self.quad_pos, self.quad_quat = pos, quat
        # self.quad_vel, self.quad_avel = vel, avel
        self.quad_vel, self.quad_avel = vel0, avel0_B
        self.quad_accel, self.quad_anaccel = accel, anaccel

    def inverse_rigid_body_dynamics(self, m, g, J, pos, quat, vel, anvel, accel, anaccel, force_body_frame=True):
        R = Rotation.from_quat(quat)
        F_world = m*(accel + np.array([0, 0, -g]))
        F_body = R.inv().apply(F_world)
        tau_body = J@anaccel + np.cross(anvel, J@anvel)
        if (force_body_frame):
            return F_body, tau_body
        return F_world, tau_body

    def forward_rigid_body_dynamics(self, m, g, J, pos, quat, vel, anvel, F, tau, force_body_frame=True):
        R = Rotation.from_quat(quat)
        if (force_body_frame):
            F_world = R.apply(F)
        else:
            F_world = F
        accel = F_world/m + np.array([0, 0, -g])
        anaccel = np.linalg.inv(J)@(tau - np.cross(anvel, J@anvel))
        return accel, anaccel

    def adjust_visualization(self):
        p.resetDebugVisualizerCamera(cameraDistance=self.view_distance, cameraYaw=self.view_yaw,
                                     cameraPitch=self.view_pitch, cameraTargetPosition=self.quad_pos)  # fix camera onto model

    def fill_model_error(self):
        F_model, tau_model = self.inverse_rigid_body_dynamics(self.M, self.G, self.J, self.quad_pos,
                                                              self.quad_quat, self.quad_vel, self.quad_avel, self.quad_accel, self.quad_anaccel)
        F_ff, tau_ff = self.inverse_rigid_body_dynamics(self.M, self.G, self.J, self.quad_pos, self.quad_quat,
                                                        self.quad_vel, self.quad_avel, self.ff_accel, self.ff_anaccel)
        self.model_error.force_body = np.array(F_model - F_ff, dtype=np.float32)
        self.model_error.torque_body = np.array(tau_model - tau_ff, dtype=np.float32)
        self.model_error.accel_world = np.array(self.quad_accel - self.ff_accel, dtype=np.float32)
        self.model_error.anaccel_body = np.array(self.quad_anaccel, dtype=np.float32)

        rot = Rotation.from_quat(self.quad_quat)
        self.model_error.force_world = np.array(rot.apply(F_model) - rot.apply(F_ff), dtype=np.float32)
        self.model_error.accel_body = np.array(rot.inv().apply(self.quad_accel) - rot.inv().apply(self.ff_accel), dtype=np.float32)

    def fill_State(self):
        self.state.header.stamp = self.get_clock().now().to_msg()
        self.state.state.pose.position = Point(x=self.quad_pos[0], y=self.quad_pos[1], z=self.quad_pos[2])
        self.state.state.pose.orientation = Quaternion(x=self.quad_quat[0], y=self.quad_quat[1], z=self.quad_quat[2], w=self.quad_quat[3])
        self.state.state.twist.linear = Vector3(x=self.quad_vel[0], y=self.quad_vel[1], z=self.quad_vel[2])
        self.state.state.twist.angular = Vector3(x=self.quad_avel[0], y=self.quad_avel[1], z=self.quad_avel[2])
        self.state.state.accel.linear = Vector3(x=self.quad_accel[0], y=self.quad_accel[1], z=self.quad_accel[2])
        self.state.state.accel.angular = Vector3(x=self.quad_anaccel[0], y=self.quad_anaccel[1], z=self.quad_anaccel[2])

    def simulation_step_callback(self):
        if (self.use_ff_state):
            self.apply_ff_state()

        self.apply_forces_torques()

        self.apply_simulation_step()

        if (self.view_follow):
            self.adjust_visualization()
        self.fill_State()
        self.fill_model_error()

    def publish_state_callback(self):
        self.state_publisher.publish(self.state)
        self.model_error_publisher.publish(self.model_error)

    def publish_image_callback(self):
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
    rclpy.init(args=args)

    node = QuadrotorPybulletDataset()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
