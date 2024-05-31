from geometry_msgs.msg import PoseStamped, TwistStamped
from numpy import sign
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu
from mocap4r2_msgs.msg import RigidBodies, RigidBody
from message_filters import ApproximateTimeSynchronizer, Subscriber
from quadrotor_interfaces.msg import State, RotorCommand
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist, Vector3, Quaternion
from scipy import signal


class LPF:
    def __init__(self):
        sample_rate = 100  # Hz
        cutoff = 10  # Hz
        self.b = signal.firwin(10, 10, fs= 200)
        self.data = 0
        self.z = signal.lfilter_zi(self.b, 1) * self.data

    def update(self, data):
        self.data = data
        self.data, self.z = signal.lfilter(self.b, 1, [self.data], zi=self.z)
        self.data = self.data[0]
        return self.data


class QuadrotorArDrone(Node):
    def __init__(self):
        super().__init__('quadrotor_ardrone_mocap', parameter_overrides=[])
        self.sub_imu = Subscriber(self, Imu, '/ardrone/imu')
        self.sub_opti = Subscriber(self, RigidBodies, '/rigid_bodies')
        self.sub_sync = ApproximateTimeSynchronizer([self.sub_imu, self.sub_opti],
                                                    10,
                                                    0.01,
                                                    )
        self.sub_sync.registerCallback(self.callback_sync)
        # self.sub_imu.registerCallback(self.callback_imu)
        # self.sub_opti.registerCallback(self.callback_mocap)

        self.pub_state = self.create_publisher(State, 'quadrotor_state', 10)
        self.pub_pose = self.create_publisher(PoseStamped, 'quadrotor_pose', 10)
        self.pub_twist = self.create_publisher(
            TwistStamped, 'quadrotor_twist', 10)
        self.pub_opti_odom = self.create_publisher(
            Odometry, 'fuse/odometry', 10)
        self.pub_imu = self.create_publisher(Imu, 'fuse/imu', 10)

        self.last_time = None
        self.last_position = [0.0, 0.0, 0.0]
        self.filter_twist_x = LPF()
        self.filter_twist_y = LPF()
        self.filter_twist_z = LPF()

    # def callback_imu(self, imu: Imu):
    #     msg = imu
    #     msg.header.frame_id = 'base_link'
    #     # imu_quat = [imu.orientation.x, imu.orientation.y,
    #     #             imu.orientation.z, imu.orientation.w]
    #     # imu_euler = Rotation.from_quat(imu_quat).as_euler('xyz', degrees=True)
    #     # imu_euler[2] = 0
    #     # imu_quat = Rotation.from_euler('xyz', imu_euler, degrees=True).as_quat()
    #     # msg.orientation.x = imu_quat[0]
    #     # msg.orientation.y = imu_quat[1]
    #     # msg.orientation.z = imu_quat[2]
    #     # msg.orientation.w = imu_quat[3]
    #     msg.orientation_covariance *= 0.0
    #     msg.angular_velocity_covariance *= 0.0
    #     msg.linear_acceleration_covariance *= 0.0
    #     self.pub_imu.publish(msg)
    #
    def callback_mocap(self, opti: RigidBodies):
        rb: RigidBody = opti.rigidbodies[0]
        # msg = Odometry()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = 'odom'
        # msg.child_frame_id = 'base_link'
        # msg.pose.pose = rb.pose
        # self.pub_opti_odom.publish(msg)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose = rb.pose
        self.pub_pose.publish(msg)

    def callback_sync(self, imu: Imu, opti: RigidBodies):
        imu_quat = [imu.orientation.x, imu.orientation.y,
                    imu.orientation.z, imu.orientation.w]
        imu_euler = Rotation.from_quat(imu_quat).as_euler('xyz', degrees=True)

        rb: RigidBody = opti.rigidbodies[0]
        rb_quat = [rb.pose.orientation.x, rb.pose.orientation.y,
                   rb.pose.orientation.z, rb.pose.orientation.w]
        rb_euler = Rotation.from_quat(rb_quat).as_euler('xyz', degrees=True)

        # fuse
        quat = Rotation.from_euler(
            'xyz', [imu_euler[0], imu_euler[1], rb_euler[2],], degrees=True).as_quat()

        position = [rb.pose.position.x, rb.pose.position.y, rb.pose.position.z]
        velocity = [0.0, 0.0, 0.0]
        if self.last_time is not None:
            dt = (self.get_clock().now() - self.last_time).nanoseconds / 1e9
            velocity = [(position[0] - self.last_position[0]) / dt,
                        (position[1] - self.last_position[1]) / dt,
                        (position[2] - self.last_position[2]) / dt]
        self.last_position = position
        self.last_time = self.get_clock().now()

        angular_velocity = [imu.angular_velocity.x,
                            imu.angular_velocity.y, imu.angular_velocity.z]
        velocity[0] = self.filter_twist_x.update(velocity[0])
        velocity[1] = self.filter_twist_y.update(velocity[1])
        velocity[2] = self.filter_twist_z.update(velocity[2])
        state = State()
        state.header.stamp = self.get_clock().now().to_msg()
        state.state.pose.orientation.x = quat[0]
        state.state.pose.orientation.y = quat[1]
        state.state.pose.orientation.z = quat[2]
        state.state.pose.orientation.w = quat[3]

        state.state.pose.position.x = position[0]
        state.state.pose.position.y = position[1]
        state.state.pose.position.z = position[2]

        state.state.twist.linear.x = velocity[0]
        state.state.twist.linear.y = velocity[1]
        state.state.twist.linear.z = velocity[2]

        state.state.twist.angular.x = angular_velocity[0]
        state.state.twist.angular.y = angular_velocity[1]
        state.state.twist.angular.z = angular_velocity[2]

        pose_s = PoseStamped()
        pose_s.header.frame_id = 'odom'
        pose_s.pose = state.state.pose

        twist_s = TwistStamped()
        twist_s.header.frame_id = 'odom'
        twist_s.twist = state.state.twist

        self.pub_state.publish(state)
        self.pub_pose.publish(pose_s)
        self.pub_twist.publish(twist_s)


def main():
    rclpy.init()
    node = QuadrotorArDrone()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
