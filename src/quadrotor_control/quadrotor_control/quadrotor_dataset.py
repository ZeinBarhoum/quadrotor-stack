import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import RotorCommand, State

import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.transform import Rotation as R
from glob import glob
import os
import time
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

DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorDataset(Node):
    def __init__(self):
        super().__init__('quadorotor_dataset_node')

        self.declare_parameters(parameters=[("dataset_package", 'quadrotor_control'),
                                            ("dataset_folder", 'resource/datasets/NeuroBEM'),
                                            ("dataset_name", 'merged_2021-02-03-16-54-28_'),
                                            ('time_field', 't'),
                                            ('command_fields', ['mot 1', 'mot 2', 'mot 3', 'mot 4']),
                                            ('state_fields', ['pos x', 'pos y', 'pos z',
                                                              'quat x', 'quat y', 'quat z', 'quat w',
                                                              'vel x', 'vel y', 'vel z',
                                                              'ang vel x', 'ang vel y', 'ang vel z',
                                                              'acc x', 'acc y', 'acc z',
                                                              'ang acc x', 'ang acc y', 'ang acc z']),
                                            ('state_frames', ['world', 'world', 'world', 'world', 'world', 'world']),
                                            ('including_gravity', True),
                                            ("interpolation_method", 'linear'),
                                            ('rotor_speeds_topic', 'quadrotor_rotor_speeds'),
                                            ('state_topic', 'quadrotor_ff_state'),
                                            ('command_publish_frequency', DEFAULT_FREQUENCY),
                                            ],
                                namespace='')

        self.dataset_package = self.get_parameter('dataset_package').get_parameter_value().string_value
        self.dataset_folder = self.get_parameter('dataset_folder').get_parameter_value().string_value
        self.dataset_name = self.get_parameter('dataset_name').get_parameter_value().string_value
        self.time_field = self.get_parameter('time_field').get_parameter_value().string_value
        self.command_fields = self.get_parameter('command_fields').get_parameter_value().string_array_value
        self.state_fields = self.get_parameter('state_fields').get_parameter_value().string_array_value
        self.state_frames = self.get_parameter('state_frames').get_parameter_value().string_array_value
        self.including_gravity = self.get_parameter('including_gravity').get_parameter_value().bool_value
        self.interpolation_method = self.get_parameter('interpolation_method').get_parameter_value().string_value
        self.rotor_speeds_topic = self.get_parameter('rotor_speeds_topic').get_parameter_value().string_value
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self.command_publish_frequency = self.get_parameter('command_publish_frequency').get_parameter_value().integer_value

        # Initialize Dataset LUT
        self.LUT_command: sp.interpolate.interp1d = None
        self.LUT_state: sp.interpolate.interp1d = None
        self.t_max = None
        self.initialize_dataset()
        # time.sleep(2)
        # Initialize Publisher
        self.command_publisher = self.create_publisher(msg_type=RotorCommand,
                                                       topic=self.rotor_speeds_topic,
                                                       qos_profile=DEFAULT_QOS_PROFILE)
        self.state_publisher = self.create_publisher(msg_type=State,
                                                     topic=self.state_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)
        # Initialize timers
        self.publisher_period = 1.0 / self.command_publish_frequency
        self.publisher_timer = self.create_timer(timer_period_sec=self.publisher_period,
                                                 callback=self.publish)
        # Announce that the node is initialized
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'Dataset node initialized at {self.start_time.seconds_nanoseconds()}')

    def initialize_dataset(self):
        df = pd.DataFrame()
        folder = os.path.join(get_package_share_directory(self.dataset_package), self.dataset_folder)
        files = glob(folder + '/' + self.dataset_name + '*.csv')
        files.sort()
        df_total = pd.DataFrame()
        for file in files:
            df = pd.read_csv(file)
            df_total = df_total.append(df)
        # self.get_logger().info(f"{df_total.head()}")
        t = df_total[self.time_field].to_numpy()
        commands = df_total[self.command_fields].to_numpy()
        state = df_total[self.state_fields].to_numpy()
        for i in range(len(state)):
            pos = state[i, 0:3]
            quat = state[i, 3:7]
            rot = R.from_quat(quat)
            v = state[i, 7:10]
            anvel = state[i, 10:13]
            accel = state[i, 13:16]
            anaccel = state[i, 16:19]

            # ensure that pos is position of o_b in W
            if self.state_frames[0] == 'body':
                pos = -pos
            # ensure that statisfies p_w = R * p_b
            if self.state_frames[1] == 'body':
                rot = rot.inv()
            # ensure that v is in W
            if self.state_frames[2] == 'body':
                v = rot.apply(v)
            # ensure that anvel is in B
            if self.state_frames[3] == 'world':
                anvel = rot.inv().apply(anvel)
            # ensure that accel is in W
            if self.state_frames[4] == 'body':
                accel = rot.apply(accel)
            # ensure that anaccel is in B
            if self.state_frames[5] == 'world':
                anaccel = rot.inv().apply(anaccel)
            # remove the gravity acceleration
            if self.including_gravity:
                accel[2] = accel[2] - 9.81

            state[i, 0:3] = pos
            state[i, 3:7] = rot.as_quat()
            state[i, 7:10] = v
            state[i, 10:13] = anvel
            state[i, 13:16] = accel
            state[i, 16:19] = anaccel

        if (t[0] > 0):
            t = t - t[0]
        self.LUT_command = sp.interpolate.interp1d(t, commands, axis=0, kind=self.interpolation_method)
        self.LUT_state = sp.interpolate.interp1d(t, state, axis=0, kind=self.interpolation_method)
        self.t_max = t[-1]

    def publish(self):
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if t > self.t_max:
            self.get_logger().info('Finished publishing commands')
            self.publisher_timer.cancel()
            return
        commands = self.LUT_command(t)
        state = self.LUT_state(t)

        rotor_speeds_msg = RotorCommand()
        rotor_speeds_msg.rotor_speeds = np.array(commands, dtype=np.float32)

        state_msg = State()
        state_msg.state.pose.position.x = state[0]
        state_msg.state.pose.position.y = state[1]
        state_msg.state.pose.position.z = state[2]
        state_msg.state.pose.orientation.x = state[3]
        state_msg.state.pose.orientation.y = state[4]
        state_msg.state.pose.orientation.z = state[5]
        state_msg.state.pose.orientation.w = state[6]
        state_msg.state.twist.linear.x = state[7]
        state_msg.state.twist.linear.y = state[8]
        state_msg.state.twist.linear.z = state[9]
        state_msg.state.twist.angular.x = state[10]
        state_msg.state.twist.angular.y = state[11]
        state_msg.state.twist.angular.z = state[12]
        state_msg.state.accel.linear.x = state[13]
        state_msg.state.accel.linear.y = state[14]
        state_msg.state.accel.linear.z = state[15]
        state_msg.state.accel.angular.x = state[16]
        state_msg.state.accel.angular.y = state[17]
        state_msg.state.accel.angular.z = state[18]

        self.command_publisher.publish(rotor_speeds_msg)
        self.state_publisher.publish(state_msg)
        # self.get_logger().info(f"Publishing command: {rotor_speeds_msg.rotor_speeds}")


def main():
    rclpy.init()
    node = QuadrotorDataset()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
