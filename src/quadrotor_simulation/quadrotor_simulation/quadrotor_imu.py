import numpy as np
from quadrotor_interfaces.msg import State
from sensor_msgs.msg import Imu
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

DEFAULT_FREQUENCY = 500.0  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorIMU(Node):
    def __init__(self):
        super().__init__('quadrotor_imu_node')

        # Declare the parameters
        self.declare_parameters(namespace='', parameters=[('imu_topic', 'quadrotor_imu'),
                                                          ('state_topic', 'quadrotor_state'),
                                                          ('imu_publishing_frequency', DEFAULT_FREQUENCY),
                                                          ('include_gravity', True),
                                                          ('angular_velocity_mean', [0.0, 0.0, 0.0]),
                                                          ('linear_acceleration_mean', [0.0, 0.0, 0.0]),
                                                          ('angular_velocity_covariance', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                                          ('linear_acceleration_covariance', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])

        # Get the parameters
        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self.imu_publishing_frequency = self.get_parameter('imu_publishing_frequency').get_parameter_value().double_value
        self.include_gravity = self.get_parameter('include_gravity').get_parameter_value().bool_value
        self.angular_velocity_mean = self.get_parameter('angular_velocity_mean').get_parameter_value().double_array_value
        self.linear_acceleration_mean = self.get_parameter('linear_acceleration_mean').get_parameter_value().double_array_value
        self.angular_velocity_covariance = self.get_parameter('angular_velocity_covariance').get_parameter_value().double_array_value
        self.linear_acceleration_covariance = self.get_parameter('linear_acceleration_covariance').get_parameter_value().double_array_value

        # Create publishers and subscribers
        self.state_subscriber = self.create_subscription(msg_type=State,
                                                         topic=self.state_topic,
                                                         callback=self.receive_state_callback,
                                                         qos_profile=DEFAULT_QOS_PROFILE)
        self.imu_publisher = self.create_publisher(msg_type=Imu,
                                                   topic=self.imu_topic,
                                                   qos_profile=DEFAULT_QOS_PROFILE)

        # Control the frequency of publishing
        self.imu_publisher_period = 1.0 / self.imu_publishing_frequency

        # Initialize the published and received data
        self.initialize_data()

        # Initialize timers
        self.imu_publishing_timer = self.create_timer(timer_period_sec=self.imu_publisher_period,
                                                      callback=self.publish_imu_callback)

        # Announce the initialization
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'QuadrotorIMU node initialized at {self.start_time.seconds_nanoseconds()}')

    def initialize_data(self):
        self.state = State()
        self.imu = Imu()
        self.imu.angular_velocity_covariance = np.array(self.angular_velocity_covariance, dtype=np.float64)
        self.imu.linear_acceleration_covariance = np.array(self.linear_acceleration_covariance, dtype= np.float64)

    def receive_state_callback(self, msg: State):
        self.state = msg

    def publish_imu_callback(self):
        w_act = np.array([self.state.state.twist.angular.x, self.state.state.twist.angular.y, self.state.state.twist.angular.z])
        w_noise_mean = np.array(self.angular_velocity_mean)
        w_noise_cov = np.array(self.angular_velocity_covariance).reshape((3, 3))
        w_noise = np.random.multivariate_normal(w_noise_mean, w_noise_cov)
        w_imu = w_act.flatten() + w_noise.flatten()

        a_W_act = np.array([self.state.state.accel.linear.x, self.state.state.accel.linear.y, self.state.state.accel.linear.z])
        if self.include_gravity:
            a_W_act[2] += 9.81  # Add gravity
        q_act = np.array([self.state.state.pose.orientation.x, self.state.state.pose.orientation.y,
                         self.state.state.pose.orientation.z, self.state.state.pose.orientation.w])
        rot = Rotation.from_quat(q_act)
        a_B_act = rot.inv().apply(a_W_act)
        a_noise_mean = np.array(self.linear_acceleration_mean)
        a_noise_cov = np.array(self.linear_acceleration_covariance).reshape((3, 3))
        a_noise = np.random.multivariate_normal(a_noise_mean, a_noise_cov)
        a_imu = a_B_act.flatten() + a_noise.flatten()

        self.imu.header.stamp = self.get_clock().now().to_msg()
        self.imu.angular_velocity.x = w_imu[0]
        self.imu.angular_velocity.y = w_imu[1]
        self.imu.angular_velocity.z = w_imu[2]
        self.imu.linear_acceleration.x = a_imu[0]
        self.imu.linear_acceleration.y = a_imu[1]
        self.imu.linear_acceleration.z = a_imu[2]
        self.imu_publisher.publish(self.imu)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorIMU()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
