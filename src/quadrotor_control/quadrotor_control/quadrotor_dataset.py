import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import RotorCommand

import numpy as np
import pandas as pd
import scipy as sp
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
                                            ("interpolation_method", 'linear'),
                                            ('rotor_speeds_topic', 'quadrotor_rotor_speeds'),
                                            ('command_publish_frequency', DEFAULT_FREQUENCY),
                                            ],
                                namespace='')

        self.dataset_package = self.get_parameter('dataset_package').get_parameter_value().string_value
        self.dataset_folder = self.get_parameter('dataset_folder').get_parameter_value().string_value
        self.dataset_name = self.get_parameter('dataset_name').get_parameter_value().string_value
        self.time_field = self.get_parameter('time_field').get_parameter_value().string_value
        self.command_fields = self.get_parameter('command_fields').get_parameter_value().string_array_value
        self.interpolation_method = self.get_parameter('interpolation_method').get_parameter_value().string_value
        self.rotor_speeds_topic = self.get_parameter('rotor_speeds_topic').get_parameter_value().string_value
        self.command_publish_frequency = self.get_parameter('command_publish_frequency').get_parameter_value().integer_value

        # Initialize Dataset LUT
        self.LUT: sp.interpolate.interp1d = None
        self.t_max = None
        self.initialize_dataset()
        time.sleep(2)
        # Initialize Publisher
        self.command_publisher = self.create_publisher(msg_type=RotorCommand,
                                                       topic=self.rotor_speeds_topic,
                                                       qos_profile=DEFAULT_QOS_PROFILE)

        # Initialize timers
        self.command_publishing_period = 1.0 / self.command_publish_frequency
        self.command_publishing_timer = self.create_timer(timer_period_sec=self.command_publishing_period,
                                                          callback=self.publish_command)
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
        if (t[0] > 0):
            t = t - t[0]
        self.LUT = sp.interpolate.interp1d(t, commands, axis=0, kind=self.interpolation_method)
        self.t_max = t[-1]

    def publish_command(self):
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if t > self.t_max:
            self.get_logger().info('Finished publishing commands')
            self.command_publishing_timer.cancel()
            return
        commands = self.LUT(t)
        rotor_command = RotorCommand()
        rotor_command.rotor_speeds = np.array(commands, dtype=np.float32)
        self.command_publisher.publish(rotor_command)
        self.get_logger().info(f"Publishing command: {rotor_command.rotor_speeds}")


def main():
    rclpy.init()
    node = QuadrotorDataset()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
