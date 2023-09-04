import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import PolynomialTrajectory, PolynomialSegment, ReferenceState
import numpy as np

# For colored traceback
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()


class QuadrotorReferencePublisher(Node):
    def __init__(self):
        super().__init__('quadrotor_reference_publisher')

        self.subscriber_poly = self.create_subscription(
            PolynomialTrajectory,
            'quadrotor_polynomial_trajectory',
            self.receive_poly_trajectory_callback,
            10  # Queue size
        )

        self.publisher_ref = self.create_publisher(
            ReferenceState,
            'quadrotor_reference',
            10  # Queue size
        )

        self.poly_x = np.array([0])
        self.poly_y = np.array([0])
        self.poly_z = np.array([1])
        self.t_clip = -1

        self.current_time = 0.0

        # Control the publishing rate
        self.publish_rate = 100  # Hz
        self.DT = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(self.DT, self.publish_reference)

        self.get_logger().info('Reference publisher node initialized')

    def publish_reference(self):
        zero_vel = False
        if (self.t_clip > 0 and self.current_time > self.t_clip):
            self.current_time = self.t_clip
            zero_vel = True

        msg = ReferenceState()
        msg.current_state.pose.position.x = np.polyval(self.poly_x, self.current_time)
        msg.current_state.pose.position.y = np.polyval(self.poly_y, self.current_time)
        msg.current_state.pose.position.z = np.polyval(self.poly_z, self.current_time)

        msg.current_state.twist.linear.x = float(np.polyval(np.polyder(self.poly_x), self.current_time))
        msg.current_state.twist.linear.y = float(np.polyval(np.polyder(self.poly_y), self.current_time))
        msg.current_state.twist.linear.z = float(np.polyval(np.polyder(self.poly_z), self.current_time))

        if (zero_vel):
            msg.current_state.twist.linear.x = 0.0
            msg.current_state.twist.linear.y = 0.0
            msg.current_state.twist.linear.z = 0.0

        self.publisher_ref.publish(msg)

        self.current_time += self.DT

    def receive_poly_trajectory_callback(self, msg: PolynomialTrajectory):
        self.poly_x = np.array(msg.segments[0].poly_x)
        self.poly_y = np.array(msg.segments[0].poly_y)
        self.poly_z = np.array(msg.segments[0].poly_z)
        self.t_clip = msg.segments[0].duration
        self.current_time = 0.0


def main():
    rclpy.init()
    node = QuadrotorReferencePublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
