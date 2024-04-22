from numpy import dtype
from pyardrone.utils.structure import numpy
from quadrotor_interfaces.msg import State, StateData, RotorCommand
from ardrone_interfaces.msg import NAVDataDemo
from rclpy.logging import rclpy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
import numpy as np
from rclpy.node import Node


class DataMapper(Node):
    def __init__(self):
        super().__init__("data_mapper_node2", parameter_overrides=[])

        sub_rotor = self.create_subscription(RotorCommand, 'quadrotor_rotor_speeds', self.rotor_callback, 10)
        self.pub = self.create_publisher(Int32MultiArray,
                                         'ardrone_control_cmd_pwm',
                                         10)

    def rotor_callback(self, msg: RotorCommand):
        m2, m1, m4, m3 = (msg.rotor_speeds - 133.1361) / (0.7244 * 1)
        m1 = numpy.clip(m1, 0, 512)
        m2 = numpy.clip(m2, 0, 512)
        m3 = numpy.clip(m3, 0, 512)
        m4 = numpy.clip(m4, 0, 512)

        m1 = int(m1)
        m2 = int(m2)
        m3 = int(m3)
        m4 = int(m4)

        msg2 = Int32MultiArray()
        msg2.data = [m1, m2, m3, m4]
        print(m1, m2, m3, m4)
        self.pub.publish(msg2)
        # print(msg.rotor_speeds)

    def destroy_node(self):
        msg2 = Int32MultiArray()
        msg2.data = [0, 0, 0, 0]
        self.pub.publish(msg2)
        print("HI")
        return super().destroy_node()


if __name__ == '__main__':
    rclpy.init()
    node = DataMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
