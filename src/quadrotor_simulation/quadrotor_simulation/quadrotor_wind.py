import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped, Vector3


def wind_sin(t):
    t = t / 1e9
    wind = np.sin(10*t) * 5
    print(wind)
    return Vector3Stamped(vector=Vector3(x=wind, y=wind, z=wind))


def main():
    rclpy.init()
    node = rclpy.create_node('quadrotor_wind')
    publisher = node.create_publisher(Vector3Stamped, 'quadrotor_wind_speed', 10)
    timer = node.create_timer(1./500, callback=lambda: publisher.publish(wind_sin(node.get_clock().now().nanoseconds)))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
