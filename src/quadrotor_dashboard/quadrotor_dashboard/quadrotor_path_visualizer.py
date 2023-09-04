import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import PathWayPoints, State, ReferenceState
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

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

        self.fig, self.ax3 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        self.ax3.set_xlabel('X')
        self.ax3.set_ylabel('Y')
        self.ax3.set_zlabel('Z')
        self.ax3.set_xlim(-10, 10)
        self.ax3.set_ylim(-10, 10)

        self.references = [[0], [0], [0]]
        self.current_reference = [0, 0, 0]

        self.states = [[0], [0], [0]]
        self.current_state = [0, 0, 0]

        self.get_logger().info('Quadrotor Path Visualizer has been started with refresh rate of {} Hz'.format(self.publish_rate))

    def waypoints_callback(self, msg):
        x = [waypoint.x for waypoint in msg.waypoints]
        y = [waypoint.y for waypoint in msg.waypoints]
        z = [waypoint.z for waypoint in msg.waypoints]
        self.ax3.scatter(x, y, z, c='b', marker='o')

    def plot_callback(self):

        self.ax3.plot(*self.references, c='r')
        self.ax3.plot(*self.states, c='g')
        # plt.legend()

        # self.ax3.scatter(*self.current_reference, c='b', marker='o')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reference_callback(self, msg: ReferenceState):

        self.references[0].append(msg.current_state.pose.position.x)
        self.references[1].append(msg.current_state.pose.position.y)
        self.references[2].append(msg.current_state.pose.position.z)

        self.current_reference = [msg.current_state.pose.position.x, msg.current_state.pose.position.y, msg.current_state.pose.position.z]

    def state_callback(self, msg: State):

        self.states[0].append(msg.state.pose.position.x)
        self.states[1].append(msg.state.pose.position.y)
        self.states[2].append(msg.state.pose.position.z)

        self.current_state = [msg.state.pose.position.x, msg.state.pose.position.y, msg.state.pose.position.z]


def main():
    rclpy.init()
    node = QuadrotorPathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__name__':
    main()
