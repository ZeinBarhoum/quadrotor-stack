""" A ROS2 that implements trajectory optimization for quadrotors """
import rclpy
from rclpy.node import Node

from quadrotor_interfaces.msg import PolynomialSegment, PolynomialTrajectory, PathWayPoints, OccupancyGrid3D
from rosidl_runtime_py.convert import message_to_ordereddict

import numpy as np
import math
import casadi as ca
import sympy as sp

from typing import List, Union, Callable, Any, Tuple

import os
import yaml
from ament_index_python.packages import get_package_share_directory

from quadrotor_utils.collision_detection import detect_collision_trajectory
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
DEFAULT_HIGH_ORDER_CONSTRAINTS = 2
DEFAULT_ADDED_POLY_ORDER = 2


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
                        # only applies for multi-segment trajectories (TODO: Now applies for one segment also)
                        ('high_order_constraints', DEFAULT_HIGH_ORDER_CONSTRAINTS),
                        ('optimize', False),  # if true, the trajectory will be optimized
                        ('added_poly_order', DEFAULT_ADDED_POLY_ORDER),  # the added orders to polynomials for optimization (only applies if optimize is true)
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
        self.high_order_constraints = self.get_parameter_value('high_order_constraints', 'int', condition_func=lambda x: x % 2 == 0)  # need to be even
        self.optimize = self.get_parameter_value('optimize', 'bool')
        self.added_poly_order = self.get_parameter_value('added_poly_order', 'int')
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

    def get_parameter_value(self, parameter_name: str, parameter_type: str, condition_func: Union[Callable[[Any], bool], None] = None) -> Union[bool, int, float, str, List[str], List[int], List[float]]:
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
        value = None
        if parameter_type == 'bool':
            value = parameter_value.bool_value
        elif parameter_type == 'int':
            value = parameter_value.integer_value
        elif parameter_type == 'float':
            value = parameter_value.double_value
        elif parameter_type == 'str':
            value = parameter_value.string_value
        elif parameter_type == 'list[str]':
            value = parameter_value.string_array_value
        elif parameter_type == 'list[float]':
            value = parameter_value.double_array_value
        elif parameter_type == 'list[int]':
            value = parameter_value.integer_array_value
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type}")

        if (condition_func and condition_func(value) is False):
            raise ValueError(f"Parameter {parameter_name} with value {value} does not satisfy the condition")
        return value

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
        # self.get_logger().info(f'{waypoints_array=}')
        if (self.time_allocation == 'distance_proportional'):
            waypoints_times = (1.0/self.avg_velocity)*np.cumsum(np.sqrt(np.sum(np.diff(waypoints_array, axis=1)**2, axis=0)))
            # ad 0 to the begining of t
            waypoints_times = np.concatenate(([0], waypoints_times))

            if (waypoints_times[-1] < 0.1):
                waypoints_times = np.arange(0, n)*self.segment_time

        elif (self.time_allocation == 'constant'):
            waypoints_times = np.arange(0, n)*self.segment_time
        elif (self.time_allocation == 'optimization'):
            raise NotImplementedError('Optimization time allocation is not implemented yet')
        else:
            raise ValueError(f"Unsupported time allocation method: {self.time_allocation}")
        # self.get_logger().info(f'{waypoints_times=}')

        if (self.one_segment):
            self.trajectory.n = 1
            segment = PolynomialSegment()
            segment.poly_x = self._calculate_polynomial_one_segment(waypoints_times, x_waypoints)
            segment.poly_y = self._calculate_polynomial_one_segment(waypoints_times, y_waypoints)
            segment.poly_z = self._calculate_polynomial_one_segment(waypoints_times, z_waypoints)
            segment.poly_yaw = self._calculate_polynomial_one_segment(waypoints_times, yaw_waypoints)
            segment.start_time = waypoints_times[0]
            segment.end_time = waypoints_times[-1]
            self.trajectory.segments = [segment]
        else:
            solution_x = self._calculate_polynomial_multiple_segments(waypoints_times, x_waypoints)
            solution_y = self._calculate_polynomial_multiple_segments(waypoints_times, y_waypoints)
            solution_z = self._calculate_polynomial_multiple_segments(waypoints_times, z_waypoints)
            solution_yaw = self._calculate_polynomial_multiple_segments(waypoints_times, yaw_waypoints)
            # self.get_logger().info(f'{solution_x=}')
            self.trajectory.n = len(solution_x)
            self.trajectory.segments = []
            for i in range(self.trajectory.n):
                segment = PolynomialSegment()
                segment.poly_x = solution_x[i]
                segment.poly_y = solution_y[i]
                segment.poly_z = solution_z[i]
                segment.poly_yaw = solution_yaw[i]
                segment.start_time = waypoints_times[i]
                segment.end_time = waypoints_times[i+1]
                self.trajectory.segments.append(segment)
        self.get_logger().info(f'{self.trajectory=}')
        self.get_logger().info(f'{detect_collision_trajectory(self.map, self.trajectory)=}')

    def _calculate_polynomial_one_segment(self, times: np.ndarray, waypoints: np.ndarray) -> List[float]:
        if self.optimize:
            return self._calculate_polynomial_one_segment_optim(times, waypoints)

        return self._calculate_polynomial_one_segment_no_optim(times, waypoints)

    def _calculate_polynomial_one_segment_no_optim(self, times: np.ndarray, waypoints: np.ndarray) -> List[float]:
        num_waypoints = len(waypoints)
        num_constraints = num_waypoints + 2
        num_params = num_constraints
        A = np.zeros((num_constraints, num_params))
        for i in range(num_waypoints):
            A[i, :] = np.array([times[i]**j for j in range(num_params)])
        A[num_waypoints, :] = np.array([j*times[0]**(j-1) if j > 0 else 0 for j in range(num_params)])
        A[-1, :] = np.array([j*times[-1]**(j-1) for j in range(num_params)])
        b = np.zeros(num_constraints)
        b[:num_waypoints] = waypoints
        b[num_waypoints:] = [0, 0]
        # self.get_logger().info(f'{A=}')
        # self.get_logger().info(f'{b=}')
        poly = np.linalg.lstsq(A, b, rcond=None)[0]
        # self.get_logger().info(f'{poly=}')
        return list(reversed(poly))

    def _calculate_polynomial_one_segment_optim(self, times: np.ndarray, waypoints: np.ndarray) -> List[float]:
        num_waypoints = len(waypoints)
        num_constraints = num_waypoints + 2
        num_params = num_constraints + self.added_poly_order  # TODO: add parameter for added coeffs

        coeffs = sp.symbols(f'c:{num_params}')
        t = sp.symbols('t')
        poly = 0
        for i in range(num_params):
            poly += coeffs[i] * t**i

        obj = sp.integrate(sp.diff(poly, t, 4) ** 2, (t, times[0], times[-1]))
        H = sp.hessian(obj, coeffs)
        H = np.array(H).astype(np.float64)

        A = np.zeros((num_constraints, num_params))
        b = np.zeros(num_constraints)

        constraints = []
        for i in range(num_waypoints):
            constraints.append(poly.subs(t, times[i]))
        constraints.append(sp.diff(poly, t, 1).subs(t, times[0]))
        constraints.append(sp.diff(poly, t, 1).subs(t, times[-1]))

        for i in range(num_constraints):
            for j in range(num_params):
                A[i, j] = sp.diff(constraints[i], coeffs[j])

        for i in range(num_waypoints):
            b[i] = waypoints[i]

        H_c = ca.DM(H)
        A_c = ca.DM(A)
        b_c = ca.DM(b)

        qp = {}
        qp['h'] = H_c.sparsity()
        qp['a'] = A_c.sparsity()

        S = ca.conic('S', 'qpoases', qp)
        r = S(h=H, a=A, lba=b_c, uba=b_c)
        coef_opt = r['x']
        coef_opt = np.array(coef_opt).reshape(-1)
        return list(reversed(coef_opt))

    def _calculate_polynomial_multiple_segments(self, times: np.ndarray, waypoints: np.ndarray) -> List[List[float]]:
        if (self.optimize):
            return self._calculate_polynomial_multiple_segments_optim(times, waypoints)

        return self._calculate_polynomial_multiple_segments_no_optim(times, waypoints)

    def _calculate_polynomial_multiple_segments_no_optim(self, times: np.ndarray, waypoints: np.ndarray) -> List[List[float]]:
        # this function construct the matrices A,B that satisfy the equation Ax = B where x is the vector of unknown parameters
        # the vector x is the concatenation of the parameters of all the polynomials in order exept for the first polynomial which is
        # forms the last parameters and not the first (for easier construction of A)

        def get_poly_param_indices(poly_index: int, num_params_per_poly_mid: int, num_params_per_poly_terminal: int, num_polys: int) -> slice:
            if (poly_index == 0):
                return slice(-num_params_per_poly_terminal, None)
            elif (poly_index == num_polys-1):
                return slice(-2*num_params_per_poly_terminal, -num_params_per_poly_terminal)
            else:
                return slice((poly_index-1)*num_params_per_poly_mid, poly_index*num_params_per_poly_mid)

        def get_poly_der_values(poly_index: int, num_params_per_poly_mid: int, num_params_per_poly_terminal: int, der_order: int, t: float) -> np.ndarray:
            num_params = num_params_per_poly_mid if poly_index not in [0, num_polys - 1] else num_params_per_poly_terminal

            return np.array([math.perm(j, der_order) * (t**(j-der_order)) if j >= der_order else 0 for j in range(num_params)])

        def add_constraint_to_A_B(A: np.ndarray,
                                  B: np.ndarray,
                                  t: float,
                                  b_value: float,
                                  constraint_index: int,
                                  constraint_der: int,
                                  poly_index: int,
                                  num_params_per_poly_mid: int,
                                  num_params_per_poly_terminal: int,
                                  num_polys: int,
                                  sign_a_values: int = 1,
                                  ) -> Tuple[np.ndarray, np.ndarray]:
            indices = get_poly_param_indices(poly_index, num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            # self.get_logger().info(f'{indices=}')
            A_values = get_poly_der_values(poly_index, num_params_per_poly_mid, num_params_per_poly_terminal, constraint_der, t)
            # self.get_logger().info(f'{A_values=}')
            A[constraint_index, indices] = A_values * sign_a_values
            B[constraint_index] = b_value
            return A, B

        num_waypoints = len(waypoints)
        # types of polynomials: terminal, middle
        num_polys = num_waypoints - 1
        num_polys_mid = max(len(waypoints) - 3, 0)
        num_polys_terminal = num_polys - num_polys_mid

        # check issue #36 for explanation on polynomial orders
        poly_order_mid = self.high_order_constraints + 1
        poly_order_terminal = (poly_order_mid * 3 - 1) // 2

        # number of parameters
        num_params_per_poly_mid = poly_order_mid + 1
        num_params_per_poly_terminal = poly_order_terminal + 1
        # self.get_logger().info(f"{num_params_per_poly_mid=}, {num_params_per_poly_terminal=}")

        # redundant for now
        num_constraints_per_poly_mid = poly_order_mid + 1
        num_constraints_per_poly_terminal = poly_order_terminal + 1

        num_params = num_params_per_poly_mid * num_polys_mid + num_params_per_poly_terminal * num_polys_terminal
        num_constraints = num_constraints_per_poly_mid * num_polys_mid + num_constraints_per_poly_terminal * num_polys_terminal

        A = np.zeros((num_constraints, num_params))
        B = np.zeros(num_constraints)

        done_constraints = 0

        # add constraints of the first waypoint
        for der_order in range(self.high_order_constraints + 1):
            b_value = 0
            if (der_order == 0):
                b_value = waypoints[0]
            A, B = add_constraint_to_A_B(A, B, times[0], b_value, done_constraints, der_order, 0,
                                         num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            done_constraints += 1
        if not done_constraints == num_constraints:  # to check one segment trajectories (2 waypoints)
            # add constraints of the last waypoint
            for der_order in range(self.high_order_constraints + 1):
                b_value = 0
                if (der_order == 0):
                    b_value = waypoints[-1]
                A, B = add_constraint_to_A_B(A, B, times[-1], b_value, done_constraints, der_order, num_polys -
                                             1, num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
                done_constraints += 1

        # add constraints of the middle waypoints
        for i in range(1, num_waypoints - 1):
            # add position constraint for two polynomials
            A, B = add_constraint_to_A_B(A, B, times[i], waypoints[i], done_constraints, 0, i - 1,
                                         num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            done_constraints += 1
            A, B = add_constraint_to_A_B(A, B, times[i], waypoints[i], done_constraints, 0, i,
                                         num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            done_constraints += 1
            for der_order in range(1, self.high_order_constraints+1):
                # add higher derivatives constraints
                A, B = add_constraint_to_A_B(A, B, times[i], 0, done_constraints, der_order, i - 1,
                                             num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
                A, B = add_constraint_to_A_B(A, B, times[i], 0, done_constraints, der_order, i,
                                             num_params_per_poly_mid, num_params_per_poly_terminal, num_polys, sign_a_values=-1)
                done_constraints += 1

        # self.get_logger().info(f'{A=}')
        # self.get_logger().info(f'{B=}')
        sol_prams = np.linalg.lstsq(A, B, rcond=None)[0]
        # self.get_logger().info(f'{sol_prams=}')
        segments = []
        for i in range(num_polys):
            if (i == 0):
                segments.append(list(reversed(sol_prams[-num_params_per_poly_terminal:])))
            elif (i == num_polys-1):
                segments.append(list(reversed(sol_prams[-2*num_params_per_poly_terminal:-num_params_per_poly_terminal])))
            else:
                segments.append(list(reversed(sol_prams[(i-1)*num_params_per_poly_mid:i*num_params_per_poly_mid])))
        # self.get_logger().info(f'{segments=}')
        return segments

    def _calculate_polynomial_multiple_segments_optim(self, times: np.ndarray, waypoints: np.ndarray) -> List[List[float]]:
        num_waypoints = len(waypoints)
        # types of polynomials: terminal, middle
        num_polys = num_waypoints - 1
        num_polys_mid = max(len(waypoints) - 3, 0)
        num_polys_terminal = num_polys - num_polys_mid

        # check issue #36 for explanation on polynomial orders
        # for parameters = constraints
        poly_order_mid = self.high_order_constraints + 1
        poly_order_terminal = (poly_order_mid * 3 - 1) // 2

        # number of parameters
        num_params_per_poly_mid = poly_order_mid + 1 + self.added_poly_order
        num_params_per_poly_terminal = poly_order_terminal + 1 + self.added_poly_order
        # number of constraints < number of parameters
        num_constraints_per_poly_mid = poly_order_mid + 1
        num_constraints_per_poly_terminal = poly_order_terminal + 1

        num_params = num_params_per_poly_mid * num_polys_mid + num_params_per_poly_terminal * num_polys_terminal
        num_constraints = num_constraints_per_poly_mid * num_polys_mid + num_constraints_per_poly_terminal * num_polys_terminal

        A = np.zeros((num_constraints, num_params))
        b = np.zeros(num_constraints)

        t = sp.symbols('t')
        coeffs = []
        coeffs_flat = []
        polys = []
        # create symbols
        for i in range(num_polys):
            if (i == 0 or i == num_polys-1):
                coeffs.append(sp.symbols(f'c{i}_:{num_params_per_poly_terminal}'))
            else:
                coeffs.append(sp.symbols(f'c{i}_:{num_params_per_poly_mid}'))
            coeffs_flat.extend(coeffs[i])
        # create polynomials
        for i in range(num_polys):
            poly = 0
            for j in range(len(coeffs[i])):
                poly += coeffs[i][j] * t**j
            polys.append(poly)
        # create objective function
        obj = 0
        for i in range(num_polys):
            obj += sp.integrate(sp.diff(polys[i], t, 4) ** 2, (t, times[i], times[i+1]))
        # calculate the matrix H
        H = sp.hessian(obj, coeffs_flat)
        H = np.array(H).astype(np.float64)

        # create the constraints
        constraints_lhs = []
        constraints_rhs = []
        for i in range(num_waypoints):
            if (i == 0):  # first waypoint
                for j in range(self.high_order_constraints+1):
                    constraints_lhs.append(sp.diff(polys[i], t, j).subs(t, times[i]))
                    constraints_rhs.append(waypoints[i] if j == 0 else 0)
            elif (i == num_waypoints-1):  # last waypoint
                for j in range(self.high_order_constraints+1):
                    constraints_lhs.append(sp.diff(polys[i-1], t, j).subs(t, times[i]))
                    constraints_rhs.append(waypoints[i] if j == 0 else 0)
            else:
                # position constraints
                constraints_lhs.append(polys[i-1].subs(t, times[i]))
                constraints_rhs.append(waypoints[i])
                constraints_lhs.append(polys[i].subs(t, times[i]))
                constraints_rhs.append(waypoints[i])
                # higher derivatives constraints
                for j in range(1, self.high_order_constraints+1):
                    constraints_lhs.append(sp.diff(polys[i-1], t, j).subs(t, times[i]) -
                                           sp.diff(polys[i], t, j).subs(t, times[i]))
                    constraints_rhs.append(0)
        for i in range(num_constraints):
            for j in range(num_params):
                A[i, j] = sp.diff(constraints_lhs[i], coeffs_flat[j])

        for i in range(num_constraints):
            b[i] = constraints_rhs[i]

        H_c = ca.DM(H)
        A_c = ca.DM(A)
        b_c = ca.DM(b)

        qp = {}
        qp['h'] = H_c.sparsity()
        qp['a'] = A_c.sparsity()
        opts = {}
        opts['printLevel'] = 'none'
        S = ca.conic('S', 'qpoases', qp, opts)
        r = S(h=H, a=A, lba=b_c, uba=b_c)
        sol_params = r['x']
        sol_params = np.array(sol_params).reshape(-1)

        segments = []
        for i in range(num_polys):
            if (i == 0):
                segments.append(list(reversed(sol_params[0:num_params_per_poly_terminal])))
            elif (i == num_polys-1):
                segments.append(list(reversed(sol_params[-num_params_per_poly_terminal:])))
            else:
                segments.append(list(reversed(sol_params[num_params_per_poly_terminal+(i-1) *
                                num_params_per_poly_mid:num_params_per_poly_terminal+i*num_params_per_poly_mid])))
        return segments


def main():
    rclpy.init()
    node = QuadrotorPolyTrajOptimizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
