from pyardrone.utils.structure import numpy
from quadrotor_interfaces.msg import State, StateData, RotorCommand
from ardrone_interfaces.msg import NAVDataDemo
from rclpy.logging import rclpy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from mocap4r2_msgs.msg import RigidBodies, RigidBody
import numpy as np
from rclpy.node import Node
import time


class DataMapper(Node):
    def __init__(self):
        super().__init__("data_mapper_node", parameter_overrides=[])

        # sub_demo = self.create_subscription(NAVDataDemo,
        #                                     'ardrone_navdata_Demo',
        #                                     self.callback,
        #                                     10
        #                                     )
        sub_rb = self.create_subscription(RigidBodies,
                                          "/rigid_bodies",
                                          self.rb_callback,
                                          10)
        self.pub = self.create_publisher(State,
                                         'quadrotor_state',
                                         10)
        self.pub2 = self.create_publisher(PoseStamped,
                                          'quadrotor_pose',
                                          10)
        self.position = np.array([0.0, 0.0, 0.0])
        self.prev_position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.prev_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.start_position = None
        self.prev_t = time.time()
        self.t = time.time()

    def rb_callback(self, msg: RigidBodies):
        self.prev_t = self.t
        self.t = time.time()
        self.dt = self.t - self.prev_t

        rb: RigidBody = msg.rigidbodies[0]
        position = np.array([rb.pose.position.x, rb.pose.position.y, rb.pose.position.z])
        if self.start_position is None:
            self.start_position = position
        self.prev_position = self.position
        self.position = position - self.start_position
        self.dp = self.position - self.prev_position
        self.v = self.dp/self.dt

        self.prev_orientation = self.orientation
        self.orientation = np.array([rb.pose.orientation.x, rb.pose.orientation.y, rb.pose.orientation.z, rb.pose.orientation.w])
        self.prev_euler = Rotation.from_quat(self.prev_orientation).as_euler('xyz', False)
        self.euler = Rotation.from_quat(self.orientation).as_euler('xyz', False)
        self.deuler = self.euler - self.prev_euler
        self.w = self.deuler/self.dt

        print(f'Position on x: {self.position[0]: 06.2f}  on y: {self.position[1]: 06.2f}  on z: {self.position[2]: 06.2f}')
        print(f'Velocity on x: {self.v[0]: 06.2f}  on y: {self.v[1]: 06.2f}  on z: {self.v[2]: 06.2f}')
        print(f'Angular  on x: {self.w[0]: 06.2f}  on y: {self.w[1]: 06.2f}  on z: {self.w[2]: 06.2f}')
        # print(f'Position on x: {self.position[0]:.2f}  on y: {self.position[1]:.2f}  on z: {self.position[2]:.2f}')
        # print(f'Position on x: {self.position[0]:.2f}  on y: {self.position[1]:.2f}  on z: {self.position[2]:.2f}')

        data = State()
        data.state.pose.orientation.x = self.orientation[0]
        data.state.pose.orientation.y = self.orientation[1]
        data.state.pose.orientation.z = self.orientation[2]
        data.state.pose.orientation.w = self.orientation[3]

        data.state.pose.position.x = self.position[0]
        data.state.pose.position.y = self.position[1]
        data.state.pose.position.z = self.position[2]

        data.state.twist.linear.x = self.v[0]
        data.state.twist.linear.y = self.v[1]
        data.state.twist.linear.z = self.v[2]

        data.state.twist.angular.x = self.w[0]
        data.state.twist.angular.y = self.w[1]
        # data.state.twist.angular.z = self.w[2]

        pose = PoseStamped()
        pose.pose = data.state.pose
        pose.header.frame_id = 'map'

        self.pub.publish(data)
        self.pub2.publish(pose)

    def callback(self, msg: NAVDataDemo):

        phi = msg.orientation.x / 1000  # deg
        theta = msg.orientation.y / 1000
        psi = msg.orientation.z / 1000
        angles = [phi, theta, psi]
        directions = 'xyz'
        order = [0, 1, 2]
        quats = Rotation.from_euler(''.join([directions[order[i]] for i in range(3)]), [angles[order[i]] for i in range(3)], degrees=True).as_quat()

        data = State()
        # data.state.pose.orientation.x = quats[0]
        # data.state.pose.orientation.y = quats[1]
        # data.state.pose.orientation.z = quats[2]
        # data.state.pose.orientation.w = quats[3]

        data.state.pose.orientation.x = self.orientation[0]
        data.state.pose.orientation.y = self.orientation[1]
        data.state.pose.orientation.z = self.orientation[2]
        data.state.pose.orientation.w = self.orientation[3]

        data.state.pose.position.x = self.position[0]
        data.state.pose.position.y = self.position[1]
        data.state.pose.position.z = self.position[2]

        pose = PoseStamped()
        pose.pose = data.state.pose
        pose.header.frame_id = 'map'

        self.pub.publish(data)
        self.pub2.publish(pose)


if __name__ == '__main__':
    try:
        rclpy.init()
        node = DataMapper()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        print("HI")
        print("B")
