""" A ROS2 node that implements a Differential Flattness Based Controller for a quadrotor. """
from numpy.core.multiarray import array
import yaml
import os
import math
from ament_index_python.packages import get_package_share_directory
import numpy as np
from quadrotor_interfaces.msg import ReferenceState, RotorCommand, State
import rclpy
from rclpy.node import Node, ParameterDescriptor
from rosidl_runtime_py.convert import message_to_ordereddict
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

import stable_baselines3 as sb3

# For colored traceback
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

# Constants
DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorDFBC(Node):

    def __init__(self):
        """ Initialize the node's parameters, subscribers, publishers and timers."""
        super().__init__(node_name='quadrotor_dfbc_node', parameter_overrides=[])

        # Declare the parameters:
        self.declare_parameters(parameters=[('quadrotor_description', 'cf2x', ParameterDescriptor()),
                                            ('state_topic', 'quadrotor_state', ParameterDescriptor()),
                                            ('reference_topic', 'quadrotor_reference', ParameterDescriptor()),
                                            ('rotor_speeds_topic', 'quadrotor_rotor_speeds', ParameterDescriptor()),
                                            ('command_publish_frequency', DEFAULT_FREQUENCY, ParameterDescriptor()),
                                            ],
                                namespace='')
        self.update_parameters()

        # Subscribers and Publishers
        self.state_subscriber = self.create_subscription(msg_type=State,
                                                         topic=self.state_topic,
                                                         callback=self.receive_state_callback,
                                                         qos_profile=DEFAULT_QOS_PROFILE)
        self.reference_subscriber = self.create_subscription(msg_type=ReferenceState,
                                                             topic=self.reference_topic,
                                                             callback=self.receive_reference_callback,
                                                             qos_profile=DEFAULT_QOS_PROFILE)
        self.command_publisher = self.create_publisher(msg_type=RotorCommand,
                                                       topic=self.rotor_speeds_topic,
                                                       qos_profile=DEFAULT_QOS_PROFILE)

        # Initialize contants, control errors and published/subscribed data
        self.initialize_model()
        self.initialize_data()
        self.initialize_constants()

        # Initialize timers
        self.command_publishing_period = 1.0 / self.command_publish_frequency
        self.command_publishing_timer = self.create_timer(timer_period_sec=self.command_publishing_period,
                                                          callback=self.publish_command)
        # Announce that the node is initialized
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'RL Control node initialized at {self.start_time.seconds_nanoseconds()}')

    def update_parameters(self):
        # Get the parameters:
        # self.KP_XYZ = self.get_parameter('KP_XYZ').get_parameter_value().double_array_value
        # self.KD_XYZ = self.get_parameter('KD_XYZ').get_parameter_value().double_array_value
        # self.KP_RPY = self.get_parameter('KP_RPY').get_parameter_value().double_array_value
        # self.KD_RPY = self.get_parameter('KD_RPY').get_parameter_value().double_array_value
        # self.Weights = self.get_parameter('Weights').get_parameter_value().double_array_value
        self.quadrotor_description = self.get_parameter('quadrotor_description').get_parameter_value().string_value
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self.reference_topic = self.get_parameter('reference_topic').get_parameter_value().string_value
        self.rotor_speeds_topic = self.get_parameter('rotor_speeds_topic').get_parameter_value().string_value
        self.command_publish_frequency = self.get_parameter('command_publish_frequency').get_parameter_value().integer_value

    def initialize_model(self):
        # TODO: change model path
        model_path = "/home/zein/Project/quadrotor-plan-control/RL/results/models/ppo_rotors_norm_no_bonus.zip"
        self.model = sb3.PPO.load(model_path)

    def initialize_constants(self):
        """
        Initializes the constants used in the quadrotor DFBC controller.

        Reads the quadrotor parameters from a YAML file located in the quadrotor_description package,
        calculates the maximum thrust, maximum RPM, maximum torque in the XY plane and maximum torque
        around the Z axis, and sets the hover RPM based on the weight of the quadrotor.

        Raises:
            FileNotFoundError: If the configuration file couldn't be found.
            yaml.YAMLError: If the configuration file couldn't be loaded due to a YAML error.
        """

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
        # self.ARM = quadrotor_params['ARM']
        self.M = quadrotor_params['M']
        self.T2W = quadrotor_params['T2W']
        self.W = self.G * self.M
        self.ROT_MAX_VEL = np.sqrt(self.T2W*self.W/(4*self.KF))
        self.MAX_RPM = self.ROT_MAX_VEL

        self.HOVER_RPM = math.sqrt(self.W / (4 * self.KF))
        self.MAX_THRUST = self.T2W * self.W
        self.MAX_RPM = math.sqrt(self.MAX_THRUST / (4 * self.KF))
        # self.MAX_TORQUE_XY = self.ARM * self.KF * self.MAX_RPM ** 2
        # self.MAX_TORQUE_Z = 2 * self.KM * self.MAX_RPM ** 2
        self.ROTOR_DIRS = quadrotor_params['ROTOR_DIRS']
        self.ARM_X = quadrotor_params['ARM_X']
        self.ARM_Y = quadrotor_params['ARM_Y']
        self.ARM_Z = quadrotor_params['ARM_Z']
        self.J = np.array(quadrotor_params['J'])

        self.get_logger().info(f'{quadrotor_params=}')

    def initialize_data(self):
        self.actual_state = message_to_ordereddict(State())
        self.reference_state = message_to_ordereddict(ReferenceState())
        self.reference_state['current_state']['pose']['position']['z'] = 1
        self.command = RotorCommand()
        self.command.rotor_speeds = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def receive_state_callback(self, msg: State):
        """ Receives the state of the quadrotor from the state subscriber and stores it in the actual_state variable.

        Args:
            msg (State): The state of the quadrotor.
        """
        self.actual_state = message_to_ordereddict(msg)

    def receive_reference_callback(self, msg: ReferenceState):
        """ Receives the reference state of the quadrotor from the reference subscriber and stores it in the reference_state variable.

        Args:
            msg (ReferenceState): The reference state of the quadrotor.
        """
        self.reference_state = message_to_ordereddict(msg)

    def publish_command(self):
        """ Publishes the rotor commands to the command publisher. """
        self.calculate_command()
        self.command_publisher.publish(self.command)

    def calculate_command(self):
        """ Calculates the rotor commands based on the actual and reference states of the quadrotor.
        """
        actual_state = self.actual_state['state']
        reference_state = self.reference_state['current_state']

        actual_position = np.array([actual_state['pose']['position']['x'],
                                    actual_state['pose']['position']['y'],
                                    actual_state['pose']['position']['z'],
                                    ])
        actual_orientation = np.array([actual_state['pose']['orientation']['x'],
                                       actual_state['pose']['orientation']['y'],
                                       actual_state['pose']['orientation']['z'],
                                       actual_state['pose']['orientation']['w'],
                                       ])
        actual_orientation_euler = Rotation.from_quat(actual_orientation).as_euler('xyz', degrees=False)

        actual_velocity = np.array([actual_state['twist']['linear']['x'],
                                    actual_state['twist']['linear']['y'],
                                    actual_state['twist']['linear']['z'],
                                    ])
        actual_angular_velocity = np.array([actual_state['twist']['angular']['x'],
                                            actual_state['twist']['angular']['y'],
                                            actual_state['twist']['angular']['z'],
                                            ])
        reference_position = np.array([reference_state['pose']['position']['x'],
                                       reference_state['pose']['position']['y'],
                                       reference_state['pose']['position']['z'],
                                       ])
        reference_orientation = np.array([reference_state['pose']['orientation']['x'],
                                          reference_state['pose']['orientation']['y'],
                                          reference_state['pose']['orientation']['z'],
                                          reference_state['pose']['orientation']['w'],
                                          ])
        reference_orientation_euler = Rotation.from_quat(reference_orientation).as_euler('xyz', degrees=False)
        reference_velocity = np.array([reference_state['twist']['linear']['x'],
                                       reference_state['twist']['linear']['y'],
                                       reference_state['twist']['linear']['z'],
                                       ])
        reference_angular_velocity = np.array([reference_state['twist']['angular']['x'],
                                               reference_state['twist']['angular']['y'],
                                               reference_state['twist']['angular']['z'],
                                               ])
        reference_linear_acceleration = np.array([reference_state['accel']['linear']['x'],
                                                  reference_state['accel']['linear']['y'],
                                                  reference_state['accel']['linear']['z'],
                                                  ])
        # reference_position = [0, 0, 1]
        # print(actual_position)

        error_position = reference_position - actual_position
        error_velocity = reference_velocity - actual_velocity
        error_orientation = reference_orientation_euler - actual_orientation_euler
        error_angular_velocity = reference_angular_velocity - actual_angular_velocity
        error_orientation = np.array([0, 0, 0, 0])

        obs = np.concatenate((-error_position, actual_orientation, -error_velocity,  actual_angular_velocity))
        # self.get_logger().info(f"{obs}")

        rotor_speeds = self.model.predict(obs)[0]
        rotor_speeds = np.array(rotor_speeds, dtype=np.float32)
        rotor_speeds *= self.MAX_RPM

        self.command.header.stamp = self.get_clock().now().to_msg()
        self.command.rotor_speeds = rotor_speeds
        # self.command.rotor_speeds = self.calculate_rotor_speeds(desired_thrust, desired_torques, Weights)

    # def calculate_rotor_speeds(self, thrust: float, torques: np.ndarray, Weights) -> np.ndarray:
    #     """ Claculate the rotor speeds using the thrust and torques.
    #     Uses the following equation:
    #         [thrust, torques] = A * [w1^2, w2^2, w3^2, w4^2]
    #
    #     Args:
    #         thrust (float): The desired thrust.
    #         torques (np.ndarray): The desired torques.
    #
    #     Returns:
    #         np.ndarray: The desired rotor speeds.
    #     """
    #     # self.get_logger().info(f'{thrust=:.2f} {torques}')
    #     # A = np.array([[self.KF, self.KF, self.KF, self.KF],
    #     #               [0, self.ARM*self.KF, 0, -self.ARM*self.KF],
    #     #               [-self.ARM*self.KF, 0, self.ARM*self.KF, 0],
    #     #               [self.KM, -self.KM, self.KM, -self.KM]])
    #     A = np.array([[self.KF, self.KF, self.KF, self.KF],
    #                   self.KF*self.ARM_Y*np.array([-1, 1, 1, -1]),
    #                   self.KF*self.ARM_X*np.array([-1, -1, 1, 1]),
    #                   [-self.ROTOR_DIRS[0]*self.KM, -self.ROTOR_DIRS[1]*self.KM, -self.ROTOR_DIRS[2]*self.KM, -self.ROTOR_DIRS[3]*self.KM]])
    #
    #     # rotor_speeds_squared = np.matmul(np.linalg.inv(A), np.array([thrust, torques[0], torques[1], torques[2]]))
    #     # rotor_speeds_squared = np.clip(rotor_speeds_squared, 0, self.MAX_RPM**2)
    #     W = np.diag(np.sqrt(Weights))
    #     # self.get_logger().info(f'{W=}')
    #     rotor_speeds_squared = lsq_linear(W@A, (W@np.array([thrust, torques[0], torques[1], torques[2]]
    #                                                        ).reshape(-1, 1)).flatten(), bounds=(0, self.MAX_RPM**2)).x
    #     # self.get_logger().info(f"{rotor_speeds_squared}")
    #     rotor_speeds = np.sqrt(rotor_speeds_squared)
    #     # actual_thrust = self.KF * np.sum(rotor_speeds_squared)
    #     # actual_torques = np.array([self.ARM * self.KF * (rotor_speeds_squared[0] - rotor_speeds_squared[2]),
    #     #                            self.ARM * self.KF * (rotor_speeds_squared[1] - rotor_speeds_squared[3]),
    #     #                            self.KM * (rotor_speeds_squared[0] - rotor_speeds_squared[1] + rotor_speeds_squared[2] - rotor_speeds_squared[3])])
    #     rotor_speeds = rotor_speeds.astype(np.float32)
    #     return rotor_speeds


def main():
    try:
        rclpy.init()
        node = QuadrotorDFBC()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print('Got clean shutdown signal exception.')
    else:
        rclpy.shutdown()

    node.destroy_node()


if __name__ == '__main__':
    main()
