""" A ROS2 that implements trajectory optimization for quadrotors """
import rclpy
from rclpy.node import Node

from quadrotor_interfaces.msg import PolynomialSegment, PolynomialTrajectory, PathWayPoints, OccupancyGrid3D
from rosidl_runtime_py.convert import message_to_ordereddict
from collections import OrderedDict

import numpy as np
import math

from typing import List, Union

import os
import yaml
from ament_index_python.packages import get_package_share_directory

# For colored traceback
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_SEGMENT_TIME = 1.0
DEFAULT_AVG_VELOCITY = 1.0
DEFAULT_QOS_PROFILE = 10


class QuadrotorPolyTrajOptimizer(Node):
    def __init__(self):
        super().__init__('quadrotor_poly_traj_optimizer')

        # Declare the parameters
        self.declare_parameters(
            namespace='',
            parameters=[('time_allocation', 'distance_proportional'),  # distance_proportional, constant or optimization
                        ('segment_time', DEFAULT_SEGMENT_TIME),  # segment time in seconds for constant time allocation
                        ('avg_velocity', DEFAULT_AVG_VELOCITY),  # average velocity in m/s for distance proportional time allocation
                        ('quadrotor_description', 'cf2x'),  # quadrotor description file name (without extension)
                        ('one_segment', True),  # if true, the trajectory will be a single segment
                        ('waypoints_topic', 'quadrotor_waypoints'),
                        ('trajectory_topic', 'quadrotor_polynomial_trajectory'),
                        ('map_topic', 'quadrotor_map')
                        ])

        # Get the parameters
        self.time_allocation = self.get_parameter_value('time_allocation', 'str')
        self.segment_time = self.get_parameter_value('segment_time', 'float')
        self.avg_velocity = self.get_parameter_value('avg_velocity', 'float')
        self.quadrotor_description = self.get_parameter_value('quadrotor_description', 'str')
        self.one_segment = self.get_parameter_value('one_segment', 'bool')
        self.waypoints_topic = self.get_parameter_value('waypoints_topic', 'str')
        self.trajecotry_topic = self.get_parameter_value('trajectory_topic', 'str')
        self.map_topic = self.get_parameter_value('map_topic', 'str')

        # Subscribers and publishers
        self.waypoints_subscriber = self.create_subscription(msg_type=PathWayPoints,
                                                             topic=self.waypoints_topic,
                                                             callback=self.receive_waypoints_callback,
                                                             qos_profile=DEFAULT_QOS_PROFILE)
        self.map_subscriber = self.create_subscription(msg_type=OccupancyGrid3D,
                                                       topic=self.map_topic,
                                                       callback=self.receive_map_callback,
                                                       qos_profile=DEFAULT_QOS_PROFILE)
        self.trajectory_publisher = self.create_publisher(msg_type=PolynomialTrajectory,
                                                          topic=self.trajecotry_topic,
                                                          qos_profile=DEFAULT_QOS_PROFILE)

        # Initialize constants and publisher/subscriber data
        self.initialize_constants()
        self.initialize_data()

        # Announce that the node is initialized
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'PolyTrajOptimizer node initialized at {self.start_time.seconds_nanoseconds()}')

    def get_parameter_value(self, parameter_name: str, parameter_type: str) -> Union[bool, int, float, str, List[str]]:
        """
        Get the value of a parameter with the given name and type.

        Args:
            parameter_name (str): The name of the parameter to retrieve.
            parameter_type (str): The type of the parameter to retrieve. Supported types are 'bool', 'int', 'float', 'str',
                'list[float]', 'list[str]' and 'list[int]'.

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
        elif parameter_type == 'list[int]':
            return parameter_value.integer_array_value
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type}")

    def initialize_constants(self):
        """
        Initializes the constants used in the quadrotor polynomial trajectory optimizer.

        Reads the quadrotor parameters from a YAML file located in the quadrotor_description package,
        calculates the constraints mainly the maximum RPM of the quadrotor

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
        self.ARM = quadrotor_params['ARM']
        self.M = quadrotor_params['M']
        self.T2W = quadrotor_params['T2W']
        self.W = self.G * self.M
        self.HOVER_RPM = math.sqrt(self.W / (4 * self.KF))
        self.MAX_THRUST = self.T2W * self.W
        self.MAX_RPM = math.sqrt(self.MAX_THRUST / (4 * self.KF))

    def initialize_data(self):
        """
        Initializes the data recevied/sent from the subsribers/publishers.
        """
        self.waypoints = message_to_ordereddict(PathWayPoints())
        self.map = message_to_ordereddict(OccupancyGrid3D())
        self.trajectory = PolynomialTrajectory()

    def receive_map_callback(self, msg: OccupancyGrid3D):
        """
        Callback function for the map subscriber.

        Args:
            msg (OccupancyGrid3D): The message received from the map subscriber.
        """

        self.map = msg

    def receive_waypoints_callback(self, msg: PathWayPoints):
        """
        Callback function for the waypoints subscriber.

        Args:
            msg (PathWayPoints): The message received from the waypoints subscriber.
        """

        self.waypoints = msg.waypoints
        self.headings = msg.heading_angles

        self.calculate_trajectory()

        self.trajectory_publisher.publish(self.trajectory)

    def calculate_trajectory(self):
        """ Calculate the trajectory from the waypoints 
        TODO: ADD COLLISSION CHECKING using the map
        """
        waypoints = np.array(self.waypoints)
        headings = np.array(self.headings)
        n = len(waypoints)
        x_waypoints = np.array([p.x for p in waypoints])
        y_waypoints = np.array([p.y for p in waypoints])
        z_waypoints = np.array([p.z for p in waypoints])
        yaw_waypoints = np.array([psi for psi in headings])

        waypoints_array: np.ndarray = np.array([x_waypoints, y_waypoints, z_waypoints])
        self.get_logger().info(f'{waypoints_array=}')
        if (self.time_allocation == 'distance_proportional'):
            waypoints_times = (1.0/self.avg_velocity)*np.cumsum(np.sqrt(np.sum(np.diff(waypoints_array, axis=1)**2, axis=0)))
            # ad 0 to the begining of t
            waypoints_times = np.concatenate(([0], waypoints_times))
        elif (self.time_allocation == 'constant'):
            waypoints_times = np.arange(0, n)*self.segment_time
        elif (self.time_allocation == 'optimization'):
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported time allocation method: {self.time_allocation}")

        if (self.one_segment):
            self.trajectory.n = 1
            segment = PolynomialSegment()
            segment.poly_x = self._calculate_polynomial(waypoints_times, x_waypoints)
            segment.poly_y = self._calculate_polynomial(waypoints_times, y_waypoints)
            segment.poly_z = self._calculate_polynomial(waypoints_times, z_waypoints)
            segment.poly_yaw = self._calculate_polynomial(waypoints_times, yaw_waypoints)
            segment.duration = waypoints_times[-1]
            self.trajectory.segments = [segment]
        else:
            raise NotImplementedError

    def _calculate_polynomial(self, t: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
        A = np.vstack([t**i for i in reversed(range(len(waypoints)))]).T
        poly = np.linalg.lstsq(A, waypoints, rcond=None)[0]
        self.get_logger().info(f'{poly=}')
        return poly.tolist()


def main():
    rclpy.init()
    node = QuadrotorPolyTrajOptimizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
