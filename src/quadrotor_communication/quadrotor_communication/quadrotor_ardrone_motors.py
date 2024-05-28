from quadrotor_interfaces.msg import RotorCommand
from rclpy.logging import rclpy
from std_msgs.msg import Int16MultiArray
from rclpy.node import Node
import numpy as np
from rclpy.signals import SignalHandlerOptions

class DataMapper(Node):
    def __init__(self):
        super().__init__("data_mapper_node2", parameter_overrides=[])

        sub_rotor = self.create_subscription(RotorCommand, 'quadrotor_rotor_speeds', self.rotor_callback, 10)
        self.pub = self.create_publisher(Int16MultiArray,
                                          'ardrone/motors',
                                          10)

    def rotor_callback(self, msg: RotorCommand):
        print(msg.rotor_speeds)
        m2, m1, m4, m3= (msg.rotor_speeds - 133.1361) / (0.7244 * 1)
        m1 = np.clip(m1, 0, 512)
        m2 = np.clip(m2, 0, 512)
        m3 = np.clip(m3, 0, 512)
        m4 = np.clip(m4, 0, 512)

        m1 = int(m1)
        m2 = int(m2)
        m3 = int(m3)
        m4 = int(m4)

        msg2 = Int16MultiArray()
        msg2.data = [m1, m2, m3, m4]
        print(m1, m2, m3, m4)
        self.pub.publish(msg2)
        # print(msg.rotor_speeds)
    def destroy_node(self):
        msg2 = Int16MultiArray()
        msg2.data = [0, 0, 0, 0]
        self.pub.publish(msg2)
        print("HI")
        return super().destroy_node()

    def shut(self):
        msg2 = Int16MultiArray()
        msg2.data = [0, 0, 0, 0]
        self.pub.publish(msg2)
        print("HI")



if __name__ == '__main__':
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = DataMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shut()
    finally:
        node.destroy_node()
        rclpy.shutdown()
