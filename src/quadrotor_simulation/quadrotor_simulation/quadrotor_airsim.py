#!/usr/bin/env python3

from quadrotor_interfaces.msg import State
import rclpy
from rclpy.node import Node, ParameterDescriptor

import airsim


try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_FREQUENCY_IMG = 30  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorAirSim(Node):

    def __init__(self, suffix='', **kwargs):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_camera_node'+suffix, **kwargs)

        # Declare parameters
        self.declare_parameters(namespace='', parameters=[('state_topic', 'quadrotor_state'+suffix, ParameterDescriptor()),])

        # Get the parameters
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value

        # Subscribers and Publishers
        self.state_subscriber = self.create_subscription(msg_type=State,
                                                         topic=self.state_topic,
                                                         callback=self.receive_state_callback,
                                                         qos_profile=DEFAULT_QOS_PROFILE)

        # Initialize the published and received data
        self.initialize_airsim_api()
        self.initialize_data()

        # initialize timers
        # Announce that the node is initialized
        self.start_time = self.get_clock().now()  # For logging purposes
        self.get_logger().info(f'QuadrotorAirSim node initialized at {self.start_time.seconds_nanoseconds()}')

    def initialize_airsim_api(self):
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        self.init_pose = self.client.simGetVehiclePose()

    def initialize_data(self):
        self.state = State()

    def receive_state_callback(self, msg):
        self.state = msg
        self.publish_airsim_pose()

    def publish_airsim_pose(self):
        pose_ros = self.state.state.pose
        position_ros = pose_ros.position
        orientation_ros = pose_ros.orientation
        # AirSim uses NED and we uses FLU; therefore, x->x, y->-y, z->-z
        position_airsim = airsim.Vector3r(position_ros.x, -position_ros.y, -position_ros.z)
        orientation_airsim = airsim.Quaternionr(orientation_ros.x, -orientation_ros.y, -orientation_ros.z, orientation_ros.w)
        pose_airsim = airsim.Pose(position_airsim, orientation_airsim)
        self.client.simSetVehiclePose(pose_airsim, True)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorAirSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print('Got clean shutdown signal exception.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
