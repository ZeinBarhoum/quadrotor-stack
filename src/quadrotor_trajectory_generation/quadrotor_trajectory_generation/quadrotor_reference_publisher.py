import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import PolynomialTrajectory, PolynomialSegment, ReferenceState, StateData

from scipy.spatial.transform import Rotation

import numpy as np

from typing import List

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

        self.n_segments = 0
        self.segments = [PolynomialSegment(poly_z=[1.0])]
        self.durations = [0.0]
        self.current_segment = 0
        self.current_time = 0.0
        self.finished = False

        self.n_future = 10
        self.dt_future = 0.1

        # Control the publishing rate
        self.publish_rate = 240  # Hz
        self.DT = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(self.DT, self.publish_reference)

        self.get_logger().info('Reference publisher node initialized')

    def publish_reference(self):
        reference_state_msg = ReferenceState()
        reference_state_msg.header.stamp = self.get_clock().now().to_msg()

        if (self.current_segment >= self.n_segments):
            self.finished = True
            segment = self.segments[self.n_segments - 1]
            t = segment.end_time
        else:
            self.finished = False
            segment = self.segments[self.current_segment]
            t = self.current_time
        reference_state_msg.current_state = self.generate_state_data_msg(poly_x=segment.poly_x,
                                                                         poly_y=segment.poly_y,
                                                                         poly_z=segment.poly_z,
                                                                         poly_yaw=segment.poly_yaw,
                                                                         t=t)
        reference_state_msg.future_states = []
        for i in range(self.n_future):
            t = self.current_time + i * self.dt_future
            if (t > segment.end_time or self.finished):
                break
            state = self.generate_state_data_msg(poly_x=segment.poly_x,
                                                 poly_y=segment.poly_y,
                                                 poly_z=segment.poly_z,
                                                 poly_yaw=segment.poly_yaw,
                                                 t=t)
            reference_state_msg.future_states.append(state)
        reference_state_msg.n = len(reference_state_msg.future_states)
        reference_state_msg.dt = self.dt_future

        self.current_time += self.DT
        if (self.current_time > segment.end_time):
            self.current_segment += 1

        self.publisher_ref.publish(reference_state_msg)

    def generate_state_data_msg(self, poly_x, poly_y, poly_z, poly_yaw, t):
        state = StateData()

        # position
        state.pose.position.x = float(np.polyval(poly_x, t))
        state.pose.position.y = float(np.polyval(poly_y, t))
        state.pose.position.z = float(np.polyval(poly_z, t))

        # orientation
        yaw = float(np.polyval(poly_yaw, t))
        orientation = Rotation.from_euler('xyz', [0.0, 0.0, yaw]).as_quat()
        state.pose.orientation.x = orientation[0]
        state.pose.orientation.y = orientation[1]
        state.pose.orientation.z = orientation[2]
        state.pose.orientation.w = orientation[3]

        # linear velocity
        state.twist.linear.x = float(np.polyval(np.polyder(poly_x), t))
        state.twist.linear.y = float(np.polyval(np.polyder(poly_y), t))
        state.twist.linear.z = float(np.polyval(np.polyder(poly_z), t))

        # angular velocity
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = float(np.polyval(np.polyder(poly_yaw), t))

        # linear acceleration
        state.accel.linear.x = (1-self.finished)*float(np.polyval(np.polyder(poly_x, 2), t))
        state.accel.linear.y = (1-self.finished)*float(np.polyval(np.polyder(poly_y, 2), t))
        state.accel.linear.z = (1-self.finished)*float(np.polyval(np.polyder(poly_z, 2), t))

        # angular acceleration
        state.accel.angular.x = 0.0
        state.accel.angular.y = 0.0
        state.accel.angular.z = (1-self.finished)*float(np.polyval(np.polyder(poly_yaw, 2), t))

        return state

    def receive_poly_trajectory_callback(self, msg: PolynomialTrajectory):
        self.n_segments = msg.n
        self.segments: List[PolynomialSegment] = msg.segments
        self.current_segment = 0
        self.current_time = 0.0


def main():
    rclpy.init()
    node = QuadrotorReferencePublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
