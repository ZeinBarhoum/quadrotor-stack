import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import ModelError

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import datetime

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()

DEFAULT_FREQUENCY = 0.5
DEFAULT_WINDOW = 10.0

export_path = '/home/zein/Project/Results/Quadrotor/ModelErrors/'


class QuadrotorModelErrorsVisualizer(Node):
    def __init__(self):
        super().__init__('quadrotor_model_errors_visualizer')
        self.declare_parameters(parameters=[('refresh_rate', DEFAULT_FREQUENCY),
                                            ('plot_window', DEFAULT_WINDOW),
                                            ('model_error_topic', 'quadrotor_model_error'),
                                            ],
                                namespace='')
        self.refresh_rate = self.get_parameter('refresh_rate').get_parameter_value().double_value
        self.plot_window = self.get_parameter('plot_window').get_parameter_value().double_value
        self.model_error_topic = self.get_parameter('model_error_topic').get_parameter_value().string_value

        self.subscription = self.create_subscription(ModelError,
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
        diag = 27
        width = diag * 1920/2203  # 2203 = sqrt(1920^2 + 1080^2)
        height = diag * 1080/2203  # 2203 = sqrt(1920^2 + 1080^2)

        self.fig3d, self.ax3d = plt.subplots(1, 1, figsize=(width, height), subplot_kw={'projection': '3d'})
        self.fig1, self.axes1 = plt.subplots(3, 3, figsize=(width, height))
        self.fig2, self.axes2 = plt.subplots(3, 3, figsize=(width, height))
        self.fig3, self.axes3 = plt.subplots(2, 2, figsize=(width, height))

        self.accel_world_axes = self.axes1[0, :]
        self.accel_body_axes = self.axes1[1, :]
        self.anaccel_body_axes = self.axes1[2, :]

        self.force_world_axes = self.axes2[0, :]
        self.force_body_axes = self.axes2[1, :]
        self.torque_body_axes = self.axes2[2, :]

        # self.input_axes = self.axes3

    def initialize_data(self):
        self.model_error = ModelError()
        self.t = []
        self.error_accel_world = []
        self.error_accel_body = []
        self.error_anaccel_body = []
        self.error_force_body = []
        self.error_force_world = []
        self.error_torque_body = []
        self.error_input = []

        self.actual_accel_world = []
        self.actual_accel_body = []
        self.actual_anaccel_body = []
        self.actual_force_body = []
        self.actual_force_world = []
        self.actual_torque_body = []
        self.actual_input = []

        self.dataset_accel_world = []
        self.dataset_accel_body = []
        self.dataset_anaccel_body = []
        self.dataset_force_body = []
        self.dataset_force_world = []
        self.dataset_torque_body = []
        self.dataset_input = []

        self.pos = []

    def model_errors_callback(self, msg):
        self.error = msg.error
        self.actual = msg.actual
        self.dataset = msg.dataset

        self.t.append((self.get_clock().now().nanoseconds - self.start_time.nanoseconds)/1e9)
        self.error_accel_world.append(list(self.error.accel_world))
        self.error_accel_body.append(list(self.error.accel_body))
        self.error_anaccel_body.append(list(self.error.anaccel_body))
        self.error_force_body.append(list(self.error.force_body))
        self.error_force_world.append(list(self.error.force_world))
        self.error_torque_body.append(list(self.error.torque_body))
        self.error_input.append(list(self.error.input))

        self.actual_accel_world.append(list(self.actual.accel_world))
        self.actual_accel_body.append(list(self.actual.accel_body))
        self.actual_anaccel_body.append(list(self.actual.anaccel_body))
        self.actual_force_body.append(list(self.actual.force_body))
        self.actual_force_world.append(list(self.actual.force_world))
        self.actual_torque_body.append(list(self.actual.torque_body))
        self.actual_input.append(list(self.actual.input))

        self.dataset_accel_world.append(list(self.dataset.accel_world))
        self.dataset_accel_body.append(list(self.dataset.accel_body))
        self.dataset_anaccel_body.append(list(self.dataset.anaccel_body))
        self.dataset_force_body.append(list(self.dataset.force_body))
        self.dataset_force_world.append(list(self.dataset.force_world))
        self.dataset_torque_body.append(list(self.dataset.torque_body))
        self.dataset_input.append(list(self.dataset.input))

        self.pos.append(list(self.dataset.position))

    def refresh_callback(self):
        self.clear_plots()

        self.plot_vector(self.accel_world_axes, r'a^W', ['x', 'y', 'z'], np.array(self.actual_accel_world), np.array(self.dataset_accel_world))
        self.plot_vector(self.force_world_axes, r'f^W', ['x', 'y', 'z'], np.array(self.actual_force_world), np.array(self.dataset_force_world))
        self.plot_vector(self.accel_body_axes, r'a^B', ['x', 'y', 'z'], np.array(self.actual_accel_body), np.array(self.dataset_accel_body))
        self.plot_vector(self.force_body_axes, r'f^B', ['x', 'y', 'z'], np.array(self.actual_force_body), np.array(self.dataset_force_body))
        self.plot_vector(self.anaccel_body_axes, r'\alpha^B', ['x', 'y', 'z'], np.array(self.actual_anaccel_body), np.array(self.dataset_anaccel_body))
        self.plot_vector(self.torque_body_axes, r'\tau^B', ['x', 'y', 'z'], np.array(self.actual_torque_body), np.array(self.dataset_torque_body))
        # self.plot_input(self.input_axes, r'u', ['1', '2', '3', '4'], np.array(self.actual_input), np.array(self.dataset_input))
        self.plot_position(self.ax3d, np.array(self.pos))
        self.update_plots()
        self.calculate_errors_stats()

    def plot_position(self, axis, data):
        data = np.array(data)
        axis.plot(data[:, 0], data[:, 1], data[:, 2], label=r'$p$')
        axis.set_xlabel('x [m]')
        axis.set_ylabel('y [m]')
        axis.set_zlabel('z [m]')

    def plot_input(self, axes, base_label, subscripts, data_actual, data_dataset=None):
        for i in range(4):
            n, m = i // 2, i % 2
            axes[n, m].set_xlim([max(0, self.t[-1] - self.plot_window), self.t[-1]])
            if (data_dataset is None):
                axes[n, m].plot(self.t, data_actual[:, i], label=r'$error {}_{}$'.format(base_label, subscripts[i]))
            else:
                axes[n, m].plot(self.t, data_actual[:, i], label=r'${}_{{{},m}}$'.format(base_label, subscripts[i]))
                axes[n, m].plot(self.t, data_dataset[:, i], label=r'${}_{{{},ds}}$'.format(base_label, subscripts[i]))

    def plot_vector(self, axes, base_label, subscripts, data_actual, data_dataset=None):
        for i in range(3):
            axes[i].set_xlim([max(0, self.t[-1] - self.plot_window), self.t[-1]])
            if (data_dataset is None):
                axes[i].plot(self.t, data_actual[:, i], label=r'$error {}_{}$'.format(base_label, subscripts[i]))
            else:
                axes[i].plot(self.t, data_actual[:, i], label=r'${}_{{{},m}}$'.format(base_label, subscripts[i]))
                axes[i].plot(self.t, data_dataset[:, i], label=r'${}_{{{},ds}}$'.format(base_label, subscripts[i]))

    def calculate_stats(self, data):
        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        mx = np.max(data, axis=0)
        mn = np.min(data, axis=0)
        rmse = np.sqrt(np.mean(data**2, axis=0))
        mae = np.mean(np.abs(data), axis=0)
        stats = np.array([mn, mean, mx, std, rmse, mae])
        return stats

    def calculate_errors_stats(self):
        df = pd.DataFrame(columns=['name', 'axis', 'min', 'mean', 'max', 'std', 'rmse', 'mae', 'unit'])
        # add info about accel_world
        stats = self.calculate_stats(self.error_accel_world)
        df.loc[len(df.index)] = ['accel_world', 'x', *self.calculate_stats(self.error_accel_world)[:, 0], 'm/s^2']
        df.loc[len(df.index)] = ['accel_world', 'y', *self.calculate_stats(self.error_accel_world)[:, 1], 'm/s^2']
        df.loc[len(df.index)] = ['accel_world', 'z', *self.calculate_stats(self.error_accel_world)[:, 2], 'm/s^2']

        # add info about accel_body
        stats = self.calculate_stats(self.error_accel_body)
        df.loc[len(df.index)] = ['accel_body', 'x', *self.calculate_stats(self.error_accel_body)[:, 0], 'm/s^2']
        df.loc[len(df.index)] = ['accel_body', 'y', *self.calculate_stats(self.error_accel_body)[:, 1], 'm/s^2']
        df.loc[len(df.index)] = ['accel_body', 'z', *self.calculate_stats(self.error_accel_body)[:, 2], 'm/s^2']

        # add info about aaaccel_body
        stats = self.calculate_stats(self.error_anaccel_body)
        df.loc[len(df.index)] = ['anaccel_body', 'x', *self.calculate_stats(self.error_anaccel_body)[:, 0], 'rad/s^2']
        df.loc[len(df.index)] = ['anaccel_body', 'y', *self.calculate_stats(self.error_anaccel_body)[:, 1], 'rad/s^2']
        df.loc[len(df.index)] = ['anaccel_body', 'z', *self.calculate_stats(self.error_anaccel_body)[:, 2], 'rad/s^2']

        # add info about force world
        stats = self.calculate_stats(self.error_force_world)
        df.loc[len(df.index)] = ['force_world', 'x', *self.calculate_stats(self.error_force_world)[:, 0], 'N']
        df.loc[len(df.index)] = ['force_world', 'y', *self.calculate_stats(self.error_force_world)[:, 1], 'N']
        df.loc[len(df.index)] = ['force_world', 'z', *self.calculate_stats(self.error_force_world)[:, 2], 'N']

        # add info about force body
        stats = self.calculate_stats(self.error_force_body)
        df.loc[len(df.index)] = ['force_body', 'x', *self.calculate_stats(self.error_force_body)[:, 0], 'N']
        df.loc[len(df.index)] = ['force_body', 'y', *self.calculate_stats(self.error_force_body)[:, 1], 'N']
        df.loc[len(df.index)] = ['force_body', 'z', *self.calculate_stats(self.error_force_body)[:, 2], 'N']

        # add info about toruqe body
        stats = self.calculate_stats(self.error_torque_body)
        df.loc[len(df.index)] = ['torque_body', 'x', *self.calculate_stats(self.error_torque_body)[:, 0], 'N*m']
        df.loc[len(df.index)] = ['torque_body', 'y', *self.calculate_stats(self.error_torque_body)[:, 1], 'N*m']
        df.loc[len(df.index)] = ['torque_body', 'z', *self.calculate_stats(self.error_torque_body)[:, 2], 'N*m']

        # add info about input
        stats = self.calculate_stats(self.error_input)
        df.loc[len(df.index)] = ['input', '1', *self.calculate_stats(self.error_input)[:, 0], 'rpm']
        df.loc[len(df.index)] = ['input', '2', *self.calculate_stats(self.error_input)[:, 1], 'rpm']
        df.loc[len(df.index)] = ['input', '3', *self.calculate_stats(self.error_input)[:, 2], 'rpm']
        df.loc[len(df.index)] = ['input', '4', *self.calculate_stats(self.error_input)[:, 3], 'rpm']

        df.to_csv(f"{export_path}stats_{datetime.datetime.now()}.csv", index=False)

        df.to_csv(f"{export_path}Latest/stats.csv", index=False)

    def clear_plots(self):
        [self.accel_body_axes[i].clear() for i in range(3)]
        [self.force_body_axes[i].clear() for i in range(3)]
        [self.accel_world_axes[i].clear() for i in range(3)]
        [self.force_world_axes[i].clear() for i in range(3)]
        [self.anaccel_body_axes[i].clear() for i in range(3)]
        [self.torque_body_axes[i].clear() for i in range(3)]
        # [self.input_axes[i, j].clear() for i in range(2) for j in range(2)]
        self.ax3d.clear()

    def update_plots(self):
        [self.accel_body_axes[i].legend() for i in range(3)]
        [self.force_body_axes[i].legend() for i in range(3)]
        [self.accel_world_axes[i].legend() for i in range(3)]
        [self.force_world_axes[i].legend() for i in range(3)]
        [self.anaccel_body_axes[i].legend() for i in range(3)]
        [self.torque_body_axes[i].legend() for i in range(3)]
        # [self.input_axes[i, j].legend() for i in range(2) for j in range(2)]
        self.ax3d.legend()

        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        # self.fig3.canvas.draw()
        # self.fig3.canvas.flush_events()

        self.fig3d.canvas.draw()
        self.fig3d.canvas.flush_events()


def main():
    rclpy.init()
    node = QuadrotorModelErrorsVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__name__':
    main()
