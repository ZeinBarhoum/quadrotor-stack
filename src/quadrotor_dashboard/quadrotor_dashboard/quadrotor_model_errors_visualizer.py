import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import ModelErrors

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FREQUENCY = 10.0
DEFAULT_WINDOW = 10.0


class QuadrotorModelErrorsVisualizer(Node):
    def __init__(self):
        super().__init__('quadrotor_model_errors_visualizer')
        self.declare_parameters(parameters=[('refresh_rate', DEFAULT_FREQUENCY),
                                            ('plot_window', DEFAULT_WINDOW),
                                            ('model_error_topic', 'quadrotor_model_errors'),
                                            ],
                                namespace='')
        self.refresh_rate = self.get_parameter('refresh_rate').get_parameter_value().double_value
        self.plot_window = self.get_parameter('plot_window').get_parameter_value().double_value
        self.model_error_topic = self.get_parameter('model_error_topic').get_parameter_value().string_value

        self.subscription = self.create_subscription(ModelErrors,
                                                     self.model_error_topic,
                                                     self.model_errors_callback,
                                                     10)

        self.initialize_data()
        self.initialize_plot()

        self.refresh_time = 1.0 / self.refresh_rate
        self.refresh_timer = self.create_timer(timer_period_sec=self.refresh_time,
                                               callback=self.refresh_callback)

        self.start_time = self.get_clock().now()

    def initialize_plot(self):
        plt.ion()
        self.num_subplots = (6, 3)
        self.fig, self.axes = plt.subplots(*self.num_subplots)
        self.accel_world_axes = self.axes[0, :]
        self.force_world_axes = self.axes[1, :]
        self.accel_body_axes = self.axes[2, :]
        self.force_body_axes = self.axes[3, :]
        self.anaccel_body_axes = self.axes[4, :]
        self.torque_body_axes = self.axes[5, :]

    def initialize_data(self):
        self.model_error = ModelErrors()
        self.t = []
        self.accel_world = []
        self.accel_body = []
        self.anaccel_body = []
        self.force_body = []
        self.force_world = []
        self.torque_body = []

    def model_errors_callback(self, msg):
        self.model_error = msg
        self.t.append((self.get_clock().now().nanoseconds - self.start_time.nanoseconds)/1e9)
        self.accel_world.append(list(self.model_error.accel_world))
        self.accel_body.append(list(self.model_error.accel_body))
        self.anaccel_body.append(list(self.model_error.anaccel_body))
        self.force_body.append(list(self.model_error.force_body))
        self.force_world.append(list(self.model_error.force_world))
        self.torque_body.append(list(self.model_error.torque_body))

    def refresh_callback(self):
        self.clear_plots()

        self.plot_vector(self.accel_world_axes, np.array(self.accel_world), r'error a^W', ['x', 'y', 'z'])
        self.plot_vector(self.force_world_axes, np.array(self.force_world), r'error F^W', ['x', 'y', 'z'])
        self.plot_vector(self.accel_body_axes, np.array(self.accel_body), r'error a^B', ['x', 'y', 'z'])
        self.plot_vector(self.force_body_axes, np.array(self.force_body), r'error F^B', ['x', 'y', 'z'])
        self.plot_vector(self.anaccel_body_axes, np.array(self.anaccel_body), r'error \alpha^B', ['x', 'y', 'z'])
        self.plot_vector(self.torque_body_axes, np.array(self.torque_body), r'error \tau^B', ['x', 'y', 'z'])

        self.update_plots()

    def plot_vector(self, axes, data, base_label, subscripts):
        for i in range(3):
            axes[i].plot(self.t, data[:, i], label=r'${}_{}$'.format(base_label, subscripts[i]))

    def clear_plots(self):
        [self.accel_body_axes[i].clear() for i in range(3)]
        [self.force_body_axes[i].clear() for i in range(3)]
        [self.accel_world_axes[i].clear() for i in range(3)]
        [self.force_world_axes[i].clear() for i in range(3)]
        [self.anaccel_body_axes[i].clear() for i in range(3)]
        [self.torque_body_axes[i].clear() for i in range(3)]

    def update_plots(self):
        [self.accel_body_axes[i].legend() for i in range(3)]
        [self.force_body_axes[i].legend() for i in range(3)]
        [self.accel_world_axes[i].legend() for i in range(3)]
        [self.force_world_axes[i].legend() for i in range(3)]
        [self.anaccel_body_axes[i].legend() for i in range(3)]
        [self.torque_body_axes[i].legend() for i in range(3)]
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = QuadrotorModelErrorsVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__name__':
    main()
