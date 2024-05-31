
from geometry_msgs.msg import PoseStamped, TwistStamped
import rclpy
from rclpy.node import Node
from mocap4r2_msgs.msg import RigidBodies, RigidBody
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseWithCovariance
from scipy import signal
from quadrotor_interfaces.msg import State
from rclpy.executors import MultiThreadedExecutor


class LPF:
    def __init__(self):
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
        self.sub = self.create_subscription(RigidBodies, '/rigid_bodies', self.callback, 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/mocap/quadrotor_pose', 10)

        self.last_time = None
        self.last_position = [0.0, 0.0, 0.0]
        self.filter_twist_x = LPF()
        self.filter_twist_y = LPF()
        self.filter_twist_z = LPF()

    #     msg = imu

    def callback(self, opti: RigidBodies):

        rb: RigidBody = opti.rigidbodies[0]
        rb_quat = [rb.pose.orientation.x, rb.pose.orientation.y,
                   rb.pose.orientation.z, rb.pose.orientation.w]
        rb_euler = Rotation.from_quat(rb_quat).as_euler('xyz', degrees=True)

        # fuse
        quat = rb_quat

        position = [rb.pose.position.x, rb.pose.position.y, rb.pose.position.z]
        velocity = [0.0, 0.0, 0.0]
        if self.last_time is not None:
            dt = (self.get_clock().now() - self.last_time).nanoseconds / 1e9
            velocity = [(position[0] - self.last_position[0]) / dt,
                        (position[1] - self.last_position[1]) / dt,
                        (position[2] - self.last_position[2]) / dt]
        self.last_position = position
        self.last_time = self.get_clock().now()

        # velocity[0] = self.filter_twist_x.update(velocity[0])
        # velocity[1] = self.filter_twist_y.update(velocity[1])
        # velocity[2] = self.filter_twist_z.update(velocity[2])
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

        pose_s = PoseStamped()
        pose_s.header.frame_id = 'odom'
        pose_s.pose = state.state.pose

        self.pub_pose.publish(pose_s)


def main():
    rclpy.init()
    node = QuadrotorArDrone()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
