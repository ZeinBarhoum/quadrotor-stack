import rclpy
from rclpy.node import Node

from quadrotor_interfaces.msg import State, RotorCommand

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import math

import os
from ament_index_python.packages import get_package_share_directory
import yaml

from typing import Union, List

DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_QOS_PROFILE = 10

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()


class QuadrotorPID(Node):
    def __init__(self):
        super().__init__('quadrotor_pid_node')

        # Declare the parameters
        self.declare_parameter('KP_Z', 40.0)  # For thrust
        self.declare_parameter('KD_Z', 5.0)  # For thrust
        self.declare_parameter('KP_XY', 1.0)  # For position
        self.declare_parameter('KD_XY', 40.0)  # For position
        self.declare_parameter('KP_RP', 5.0)  # For roll and pitch
        self.declare_parameter('KD_RP', 1.0)  # For roll and pitch
        self.declare_parameter('KP_Y', 40.0)  # For yaw
        self.declare_parameter('KD_Y', 5.0)  # For yaw
        self.declare_parameter('MAX_ROLL_PITCH', math.pi / 8)  # For roll and pitch
        self.declare_parameter('quadrotor_description', 'cf2x')  # quadrotor name (for config file)
        self.declare_parameter('state_topic', 'quadrotor_state')
        self.declare_parameter('reference_topic', 'quadrotor_reference')
        self.declare_parameter('rotor_speeds_topic', 'quadrotor_rotor_speeds')
        self.declare_parameter('command_publishing_frequency', DEFAULT_FREQUENCY)

        # Get the parameters
        self.KP_Z = self.get_parameter_value('KP_Z', 'float')
        self.KD_Z = self.get_parameter_value('KD_Z', 'float')
        self.KP_XY = self.get_parameter_value('KP_XY', 'float')
        self.KD_XY = self.get_parameter_value('KD_XY', 'float')
        self.KP_RP = self.get_parameter_value('KP_RP', 'float')
        self.KD_RP = self.get_parameter_value('KD_RP', 'float')
        self.KP_Y = self.get_parameter_value('KP_Y', 'float')
        self.KD_Y = self.get_parameter_value('KD_Y', 'float')
        self.MAX_ROLL_PITCH = self.get_parameter_value('MAX_ROLL_PITCH', 'float')
        self.quadrotor_description = self.get_parameter_value('quadrotor_description', 'str')
        self.state_topic = self.get_parameter_value('state_topic', 'str')
        self.reference_topic = self.get_parameter_value('reference_topic', 'str')
        self.rotor_speeds_topic = self.get_parameter_value('rotor_speeds_topic', 'str')
        self.command_publishing_frequency = self.get_parameter_value('command_publishing_frequency', 'int')

        # Subscribers and Publishers
        self.state_subscriber = self.create_subscription(msg_type=State,
                                                         topic=self.state_topic,
                                                         callback=self.receive_state_callback,
                                                         qos_profile=DEFAULT_QOS_PROFILE
                                                         )
        self.reference_subscriber = self.create_subscription(msg_type=State,
                                                             topic=self.reference_topic,
                                                             callback=self.receive_reference_callback,
                                                             qos_profile=DEFAULT_QOS_PROFILE
                                                             )
        self.command_publisher = self.create_publisher(msg_type=RotorCommand,
                                                       topic=self.rotor_speeds_topic,
                                                       qos_profile=DEFAULT_QOS_PROFILE
                                                       )

        # Control the publishing rate
        self.command_publishing_period = 1.0 / self.command_publishing_frequency

        # Initialize the constants,  control errors and published/subscribed data
        self.initialize_constants()
        self.initialize_errors()
        self.initialize_data()

        # Initialize timers
        self.command_publishing_timer = self.create_timer(timer_period_sec=self.command_publishing_period,
                                                          callback=self.publish_command
                                                          )
        # Announce that the node is initialized
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'PID node initialized at {self.start_time.seconds_nanoseconds()}')

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
        config_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'config')
        config_file = os.path.join(config_folder, f'{self.quadrotor_description}_params.yaml')
        with open(config_file, 'r') as stream:
            try:
                parameters = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self.get_logger().error(
                    f"Cofiguration File {config_file} Couldn't Be Loaded, Raised Error {exc}")
                parameters = dict()
        quadrotor_params = parameters[f'{self.quadrotor_description.upper()}_PARAMS']
        self.G = 9.81
        self.KF = quadrotor_params['KF']
        self.KM = quadrotor_params['KM']
        self.ARM = quadrotor_params['ARM']
        self.M = quadrotor_params['M']
        self.T2W = quadrotor_params['T2W']
        self.W = self.G * self.M
        self.HOVER_RPM = math.sqrt(self.W / (4 * self.KF))
        self.MAX_THRUST = self.T2W * self.W
        self.MAX_RPM = math.sqrt(self.MAX_THRUST / (4 * self.KF))
        self.MAX_TORQUE_XY = self.ARM * self.KF * self.MAX_RPM ** 2
        self.MAX_TORQUE_Z = 2 * self.KM * self.MAX_RPM ** 2

    def initialize_errors(self):
        self.error_p = np.zeros(3)  # position
        self.error_r = np.zeros(3)  # rotation (EULER ANGLES)
        self.error_v = np.zeros(3)  # velocity
        self.error_w = np.zeros(3)  # angular velocity (body frame)

    def initialize_data(self):
        self.actual_state = {'position': np.array([0.0, 0.0, 0.0]),
                             'orientation': Rotation.from_euler('xyz', [0.0, 0.0, 0.0]),
                             'velocity': np.array([0.0, 0.0, 0.0]),
                             'angular_velocity': np.array([0.0, 0.0, 0.0])
                             }
        self.reference_state = {'position': np.array([0.0, 0.0, 0.0]),
                                'orientation': Rotation.from_euler('xyz', [0.0, 0.0, 0.0]),
                                'velocity': np.array([0.0, 0.0, 0.0]),
                                'angular_velocity': np.array([0.0, 0.0, 0.0])
                                }
        self.command = RotorCommand()

    def receive_state_callback(self, msg: State):
        self.actual_state['position'] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.actual_state['orientation'] = Rotation.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.actual_state['velocity'] = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.actual_state['angular_velocity'] = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def receive_reference_callback(self, msg: State):
        self.reference_state['position'] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.reference_state['orientation'] = Rotation.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.reference_state['velocity'] = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.reference_state['angular_velocity'] = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def publish_command(self):
        error_p = self.reference_state['position'] - self.actual_state['position']

        error_v = self.reference_state['velocity'] - self.actual_state['velocity']
        KP_XYZ = np.array([self.KP_XY, self.KP_XY, self.KP_Z])
        KD_XYZ = np.array([self.KD_XY, self.KD_XY, self.KD_Z])
        desired_acc = np.multiply(KP_XYZ, error_p) + np.multiply(KD_XYZ, error_v)
        desired_thrust = self.M * (desired_acc[2] + self.G)
        # self.get_logger().info(f'desired_acc: {desired_acc}')
        desired_yaw = self.reference_state['orientation'].as_euler('xyz')[2]
        current_yaw = self.actual_state['orientation'].as_euler('xyz')[2]
        # self.get_logger().info(f'current state: {self.actual_state}')

        desired_roll = (1.0/self.G) * (desired_acc[0] * math.sin(desired_yaw) - desired_acc[1] * math.cos(desired_yaw))
        desired_pitch = (1.0/self.G) * (desired_acc[0] * math.cos(desired_yaw) + desired_acc[1] * math.sin(desired_yaw))
        # self.get_logger().info(f'desired_pitch: {desired_pitch}')
        desired_euler = np.array([desired_roll, desired_pitch, desired_yaw])

        desired_w = np.array([0.0, 0.0, 0.0])

        error_r = desired_euler - self.actual_state['orientation'].as_euler('xyz')
        error_w = desired_w - self.actual_state['angular_velocity']

        KP_RPY = np.array([self.KP_RP, self.KP_RP, self.KP_Y])
        KD_RPY = np.array([self.KD_RP, self.KD_RP, self.KD_Y])
        desired_torques = np.multiply(KP_RPY, error_r) + np.multiply(KD_RPY, error_w)

        # calculate the rotor speeds
        self.command.rotor_speeds = self.calculate_command(desired_thrust, desired_torques)

        # publish the command
        self.command_publisher.publish(self.command)

    def calculate_command(self, thrust: float, torques: np.ndarray) -> np.ndarray:
        # self.get_logger().info(f'{thrust=:.2f} {torques}')
        A = np.array([[self.KF, self.KF, self.KF, self.KF],
                      [0, self.ARM*self.KF, 0, -self.ARM*self.KF],
                      [-self.ARM*self.KF, 0, self.ARM*self.KF, 0],
                      [self.KM, -self.KM, self.KM, -self.KM]])
        arm_angle = math.pi / 4
        A = np.array([[self.KF, self.KF, self.KF, self.KF],
                      self.KF*self.ARM*np.array([math.cos(arm_angle), math.sin(arm_angle), -math.cos(arm_angle), -math.sin(arm_angle)]),
                      self.KF*self.ARM*np.array([-math.sin(arm_angle), math.cos(arm_angle), math.sin(arm_angle), -math.cos(arm_angle)]),
                      [-self.KM, self.KM, -self.KM, self.KM]])

        rotor_speeds_squared = np.matmul(np.linalg.inv(A), np.array([thrust, torques[0], torques[1], torques[2]]))
        rotor_speeds_squared = np.clip(rotor_speeds_squared, 0, self.MAX_RPM**2)
        rotor_speeds = np.sqrt(rotor_speeds_squared)
        actual_thrust = self.KF * np.sum(rotor_speeds_squared)
        actual_torques = np.array([self.ARM * self.KF * (rotor_speeds_squared[0] - rotor_speeds_squared[2]),
                                   self.ARM * self.KF * (rotor_speeds_squared[1] - rotor_speeds_squared[3]),
                                   self.KM * (rotor_speeds_squared[0] - rotor_speeds_squared[1] + rotor_speeds_squared[2] - rotor_speeds_squared[3])])
        return rotor_speeds.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorPID()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
