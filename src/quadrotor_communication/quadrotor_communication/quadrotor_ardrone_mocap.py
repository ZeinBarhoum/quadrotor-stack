from geometry_msgs.msg import PoseStamped, TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu 
from mocap4r2_msgs.msg import RigidBodies, RigidBody
from message_filters import ApproximateTimeSynchronizer, Subscriber
from quadrotor_interfaces.msg import State, RotorCommand
from scipy.spatial.transform import Rotation


class QuadrotorArDrone(Node):
    def __init__(self):
        super().__init__('quadrotor_ardrone', parameter_overrides=[])
        self.sub_imu = Subscriber(self, Imu, '/ardrone/imu')
        self.sub_opti = Subscriber(self, RigidBodies, '/rigid_bodies')
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_imu, self.sub_opti],
            10,
            0.01,
        )
        self.ts.registerCallback(self.callback)
        self.pub_state = self.create_publisher(State, 'quadrotor_state', 10)
        self.last_time = None
        self.last_position = [0.0, 0.0, 0.0]
        

        self.pub_pose = self.create_publisher(PoseStamped, 'quadrotor_pose', 10)
        self.pub_twist = self.create_publisher(TwistStamped, 'quadrotor_twist', 10)
        
    def callback(self, imu: Imu, opti: RigidBodies):
        imu_quat = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
        imu_euler = Rotation.from_quat(imu_quat).as_euler('xyz', degrees=True)
        
        rb:RigidBody = opti.rigidbodies[0]
        rb_quat = [rb.pose.orientation.x, rb.pose.orientation.y, rb.pose.orientation.z, rb.pose.orientation.w]
        rb_euler = Rotation.from_quat(rb_quat).as_euler('xyz', degrees=True)

        # fuse
        quat = Rotation.from_euler('xyz', [imu_euler[0], imu_euler[1], rb_euler[2],], degrees=True).as_quat()

        position = [rb.pose.position.x, rb.pose.position.y, rb.pose.position.z]
        velocity = [0.0, 0.0, 0.0]
        if self.last_time is not None:
            dt = (self.get_clock().now() - self.last_time).nanoseconds / 1e9
            velocity = [(position[0] - self.last_position[0]) / dt,
                        (position[1] - self.last_position[1]) / dt,
                        (position[2] - self.last_position[2]) / dt]
        self.last_position = position
        self.last_time = self.get_clock().now()
        
        angular_velocity = [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]

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
        pose_s.header.frame_id = 'map'
        pose_s.pose = state.state.pose

        twist_s = TwistStamped()
        twist_s.header.frame_id = 'map'
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
