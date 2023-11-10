import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation

import cv2
from cv_bridge import CvBridge
import numpy as np

try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()


class QuadrotorImageVisualizer(Node):
    def __init__(self):
        super().__init__('quadrotor_image_visualizer')
        self.subscriber_image = self.create_subscription(msg_type=Image, topic='quadrotor_img',
                                                         callback=self.image_callback, qos_profile=10)

        self.start_time = self.get_clock().now()  # For logging purposes
        self.previous_time = self.start_time
        self.freqs = []
        self.get_logger().info(f'Quadrotor Image Visualizer initialized at {self.start_time.seconds_nanoseconds()}')

    def image_callback(self, msg):
        self.current_time = self.get_clock().now()
        dt = self.current_time - self.previous_time
        freq = 1 / dt.nanoseconds * 10 ** 9
        self.freqs.append(freq)

        # get image
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # show image with printed frequency
        cv2.putText(img, f'{int(np.mean(self.freqs[-100:]))} Hz', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Quadrotor Image', img)
        cv2.waitKey(1)

        self.previous_time = self.current_time


def main():
    rclpy.init()
    node = QuadrotorImageVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__name__':
    main()
