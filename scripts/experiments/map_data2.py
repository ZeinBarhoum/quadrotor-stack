import time

from ardrone_interfaces.msg import NAVDataDemo, NAVDataRawMeasures
from geometry_msgs.msg import PoseStamped, Vector3
from mocap4r2_msgs.msg import RigidBodies, RigidBody
import numpy as np
from quadrotor_interfaces.msg import State
from rclpy.logging import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from message_filters import ApproximateTimeSynchronizer, Subscriber


class DataMapper(Node):
    def __init__(self):
        super().__init__("data_mapper_node", parameter_overrides=[])
        self.sub_demo = Subscriber(self,
                                   NAVDataDemo,
                                   "ardrone_navdata_Demo",
                                   )
        self.sub_raw = Subscriber(self,
                                  NAVDataRawMeasures,
                                  "ardrone_navdata_RawMeasures",
                                  )
        self.sub_opti = Subscriber(self,
                                   RigidBodies,
                                   "rigid_bodies",
                                   )
        self.pub_pose = self.create_publisher(PoseStamped,
                                              "ardrone_pose",
                                              10)
        self.pub_state = self.create_publisher(State,
                                               "quadrotor_state",
                                               10)
        queue_size = 10
        # you can use ApproximateTimeSynchronizer if msgs dont have exactly the same timestamp
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_demo, self.sub_opti, self.sub_raw],
            queue_size,
            0.01,  # defines the delay (in seconds) with which messages can be synchronized
        )
        self.ts.registerCallback(self.callback)

        self.ts1 = ApproximateTimeSynchronizer(
            [self.sub_demo, self.sub_raw],
            queue_size,
            0.01,  # defines the delay (in seconds) with which messages can be synchronized
        )
        self.ts1.registerCallback(self.callback_no_opti)

        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.prev_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.position = np.array([0, 0, 0])
        self.prev_position = np.array([0.0, 0.0, 0.0])
        self.start_position = None
        self.prev_t = time.time()
        self.t = time.time()

    def callback_no_opti(self, demo: NAVDataDemo, raw: NAVDataRawMeasures):
        or_drone = np.array([demo.orientation.x, demo.orientation.y, demo.orientation.z])/1000  # original in melli degrees
        or_fuse = np.array([or_drone[0], 0, 0])
        or_fuse_quat = Rotation.from_euler('xyz', or_fuse, degrees=True).as_quat()

        pose_fuse = PoseStamped()
        pose_fuse.header.frame_id = 'map'
        pose_fuse.pose.orientation.x = or_fuse_quat[0]
        pose_fuse.pose.orientation.y = or_fuse_quat[1]
        pose_fuse.pose.orientation.z = or_fuse_quat[2]
        pose_fuse.pose.orientation.w = or_fuse_quat[3]

        self.w = np.array([raw.raw_imu.angular_velocity.x,
                          raw.raw_imu.angular_velocity.y,
                          raw.raw_imu.angular_velocity.z])/1000
        self.w_rad = self.w * np.pi / 180

        state = State()
        state.state.pose = pose_fuse.pose
        state.state.twist.angular = Vector3(x=self.w_rad[0], y=0.0, z=0.0)
        self.pub_state.publish(state)
        print(state)

    def callback(self, demo: NAVDataDemo, opti: RigidBodies, raw: NAVDataRawMeasures):
        self.prev_t = self.t
        self.t = time.time()
        self.dt = self.t - self.prev_t
        # self.get_logger().info("Got Message")
        or_drone = np.array([demo.orientation.x, demo.orientation.y, demo.orientation.z])/1000  # original in melli degrees

        rigid_body: RigidBody = opti.rigidbodies[0]
        or_opti_quat = np.array([rigid_body.pose.orientation.x, rigid_body.pose.orientation.y, rigid_body.pose.orientation.z, rigid_body.pose.orientation.w])

        or_opti = Rotation.from_quat(or_opti_quat).as_euler('zxy', degrees=True)
        # print(or_opti)

        or_fuse = np.array([or_drone[0], or_drone[1], or_opti[0]])
        or_fuse = np.array([or_drone[0], 0, 0])
        or_fuse_quat = Rotation.from_euler('xyz', or_fuse, degrees=True).as_quat()

        pose_fuse = PoseStamped()
        pose_fuse.header.frame_id = 'map'
        pose_fuse.pose.orientation.x = or_fuse_quat[0]
        pose_fuse.pose.orientation.y = or_fuse_quat[1]
        pose_fuse.pose.orientation.z = or_fuse_quat[2]
        pose_fuse.pose.orientation.w = or_fuse_quat[3]

        self.prev_position = self.position
        self.position = np.array([rigid_body.pose.position.x, rigid_body.pose.position.y, rigid_body.pose.position.z])
        if self.start_position is None:
            self.start_position = self.position

        self.position = self.position - self.start_position
        self.dp = self.position - self.prev_position
        self.v = self.dp/self.dt

        # pose_fuse.pose.position.x = self.position[0]
        # pose_fuse.pose.position.y = self.position[1]
        # pose_fuse.pose.position.z = self.position[2]
        # print(self.position)

        self.pub_pose.publish(pose_fuse)

        self.w = np.array([raw.raw_imu.angular_velocity.x,
                          raw.raw_imu.angular_velocity.y,
                          raw.raw_imu.angular_velocity.z])/1000
        self.w_rad = self.w * np.pi / 180
        print(f'Position    on x: {self.position[0]: 06.2f}  on y: {self.position[1]: 06.2f}  on z: {self.position[2]: 06.2f}')
        print(f'orientation on x: {or_fuse[0]: 06.2f}  on y: {or_fuse[1]: 06.2f}  on z: {or_fuse[2]: 06.2f}')
        print(f'Velocity    on x: {self.v[0]: 06.2f}  on y: {self.v[1]: 06.2f}  on z: {self.v[2]: 06.2f}')
        print(f'Angular     on x: {self.w_rad[0]: 06.2f}  on y: {self.w_rad[1]: 06.2f}  on z: {self.w_rad[2]: 06.2f}')

        state = State()
        state.state.pose = pose_fuse.pose
        # state.state.twist.linear = Vector3(x=self.v[0], y=self.v[1], z=self.v[2])
        state.state.twist.angular = Vector3(x=self.w_rad[0], y=self.w_rad[1], z=0.0)
        self.pub_state.publish(state)


if __name__ == '__main__':
    rclpy.init()
    node = DataMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
