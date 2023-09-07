import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import PathWayPoints, State, ReferenceState
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()


class QuadrotorPathVisualizer(Node):
    def __init__(self):
        super().__init__('quadrotor_dashboard')
        self.subscriber_waypoints = self.create_subscription(msg_type=PathWayPoints,
                                                             topic='quadrotor_waypoints',
                                                             callback=self.waypoints_callback,
                                                             qos_profile=10
                                                             )

        self.subscriber_referece_path = self.create_subscription(msg_type=ReferenceState,
                                                                 topic='quadrotor_reference',
                                                                 callback=self.reference_callback,
                                                                 qos_profile=10)

        self.subscriber_actual_path = self.create_subscription(msg_type=State,
                                                               topic='quadrotor_state',
                                                               callback=self.state_callback,
                                                               qos_profile=10)

        # Control the publishing rate
        self.declare_parameter('refresh_rate', 20)

        self.publish_rate = self.get_parameter('refresh_rate').get_parameter_value().integer_value

        self.DT = 1.0 / self.publish_rate  # seconds

        self.timer_plot = self.create_timer(timer_period_sec=self.DT,
                                            callback=self.plot_callback)

        plt.ion()

        self.fig = plt.figure()
        self.num_subplots = (3, 2)
        self.ax_3d = self.fig.add_subplot(*self.num_subplots, 1, projection='3d')
        self.ax_xy = self.fig.add_subplot(*self.num_subplots, 2)
        self.ax_error_pos = self.fig.add_subplot(*self.num_subplots, 3)
        self.ax_error_rot = self.fig.add_subplot(*self.num_subplots, 4)
        self.ax_yaw = self.fig.add_subplot(*self.num_subplots, 5)

        self.references = [[0], [0], [0], [0]]
        self.references_yaw = [0]
        self.current_reference = [0, 0, 0, 0]
        self.states = [[0], [0], [0], [0]]
        self.current_state = [0, 0, 0, 0]
        self.future_states = [[0], [0], [0], [0]]
        self.errors = [[0], [0], [0], [0]]
        self.current_error = [0, 0, 0, 0]
        self.start_time = -1
        self.times = [0]

        self.waypoints = [[], [], []]

        self.get_logger().info('Quadrotor Path Visualizer has been started with refresh rate of {} Hz'.format(self.publish_rate))

    def waypoints_callback(self, msg):
        x = [waypoint.x for waypoint in msg.waypoints]
        y = [waypoint.y for waypoint in msg.waypoints]
        z = [waypoint.z for waypoint in msg.waypoints]
        self.waypoints = [x, y, z]

    def plot_callback(self):
        self.ax_3d.clear()
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim(-10, 10)
        self.ax_3d.set_ylim(-10, 10)
        self.ax_3d.scatter(*self.waypoints[:3], c='b', marker='o', label='Waypoints')
        self.ax_3d.plot(*self.references[:3], c='r', label='Reference')
        self.ax_3d.plot(*self.states[:3], c='g', label='State')
        self.ax_3d.scatter(*self.future_states[:3], c='b', marker='x', label='Future States')
        self.ax_3d.legend()

        self.ax_xy.clear()
        self.ax_xy.set_xlabel('X')
        self.ax_xy.set_ylabel('Y')
        self.ax_xy.set_xlim(-10, 10)
        self.ax_xy.set_ylim(-10, 10)
        self.ax_xy.scatter(*self.waypoints[:2], c='b', marker='o', label='Waypoints')
        self.ax_xy.plot(*self.references[:2], c='r', label='Reference')
        self.ax_xy.plot(*self.states[:2], c='g', label='State')
        self.ax_xy.legend()

        self.ax_error_pos.clear()
        self.ax_error_pos.set_xlabel('Time')
        self.ax_error_pos.set_ylabel('Error')
        self.ax_error_pos.set_xlim(0, self.times[-1], auto=True)
        self.ax_error_pos.set_ylim(-1, 1, auto=False)

        self.ax_error_rot.clear()
        self.ax_error_rot.set_xlabel('Time')
        self.ax_error_rot.set_ylabel('Error')
        self.ax_error_rot.set_xlim(0, self.times[-1], auto=True)
        self.ax_error_rot.set_ylim(-1, 1, auto=False)

        self.ax_yaw.clear()
        self.ax_yaw.set_xlabel('Time')
        self.ax_yaw.set_ylabel('Yaw')
        self.ax_yaw.set_xlim(0, self.times[-1], auto=True)
        self.ax_yaw.set_ylim(-np.pi, np.pi, auto=False)
        # self.get_logger().info(f'{self.references[3]=}')
        self.ax_yaw.plot(self.references_yaw, c='r', label='Reference')
        self.ax_yaw.plot(self.states[3], c='g', label='State')
        self.ax_yaw.legend()

        num_points = 50
        if (len(self.times) > num_points):
            times = self.times[-num_points:]
            errors_x = self.errors[0][-num_points:]
            errors_y = self.errors[1][-num_points:]
            errors_z = self.errors[2][-num_points:]
            errors_yaw = self.errors[3][-num_points:]
        else:
            times = self.times
            errors_x = self.errors[0]
            errors_y = self.errors[1]
            errors_z = self.errors[2]
            errors_yaw = self.errors[3]

        self.ax_error_pos.plot(times, errors_x, c='r', label='X')
        self.ax_error_pos.plot(times, errors_y, c='g', label='Y')
        self.ax_error_pos.plot(times, errors_z, c='b', label='Z')
        self.ax_error_pos.legend()

        self.ax_error_rot.plot(times, errors_yaw, c='r', label='Yaw')
        self.ax_error_rot.legend()

        # plt.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reference_callback(self, msg: ReferenceState):

        self.references[0].append(msg.current_state.pose.position.x)
        self.references[1].append(msg.current_state.pose.position.y)
        self.references[2].append(msg.current_state.pose.position.z)
        quat = np.array([msg.current_state.pose.orientation.x, msg.current_state.pose.orientation.y,
                        msg.current_state.pose.orientation.z, msg.current_state.pose.orientation.w])
        euler = Rotation.from_quat(quat).as_euler('xyz')
        self.references[3].append(euler[2])

        self.current_reference = [msg.current_state.pose.position.x, msg.current_state.pose.position.y, msg.current_state.pose.position.z, euler[2]]

        num_future = msg.n
        self.future_states[0] = [msg.future_states[i].pose.position.x for i in range(num_future)]
        self.future_states[1] = [msg.future_states[i].pose.position.y for i in range(num_future)]
        self.future_states[2] = [msg.future_states[i].pose.position.z for i in range(num_future)]

    def state_callback(self, msg: State):

        self.states[0].append(msg.state.pose.position.x)
        self.states[1].append(msg.state.pose.position.y)
        self.states[2].append(msg.state.pose.position.z)

        quat = np.array([msg.state.pose.orientation.x, msg.state.pose.orientation.y,
                        msg.state.pose.orientation.z, msg.state.pose.orientation.w])
        euler = Rotation.from_quat(quat).as_euler('xyz')
        self.states[3].append(euler[2])
        self.references_yaw.append(self.references[3][-1])

        self.current_state = [msg.state.pose.position.x, msg.state.pose.position.y, msg.state.pose.position.z, euler[2]]

        self.current_error = list(np.array(self.current_reference) - np.array(self.current_state))

        self.errors[0].append(self.current_error[0])
        self.errors[1].append(self.current_error[1])
        self.errors[2].append(self.current_error[2])
        self.errors[3].append(self.current_error[3])

        self.times.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if (self.start_time == -1):
            self.start_time = self.times[-1]

        self.times[-1] -= self.start_time


def main():
    rclpy.init()
    node = QuadrotorPathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__name__':
    main()
