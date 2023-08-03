import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import PolynomialTrajectory, PathWayPoints
from std_msgs.msg import Float32MultiArray
import numpy as np


class QuadrotorPolyOptimizer(Node):
    def __init__(self):
        super().__init__('quadrotor_poly_optimizer')

        self.subscriber_waypoiny = self.create_subscription(
            PathWayPoints,
            'quadrotor_waypoints',
            self.receive_waypoints_callback,
            10  # Queue size
        )

        self.publisher = self.create_publisher(
            PolynomialTrajectory,
            'quadrotor_polynomial_trajectory',
            10 # Queue size
        )

        self.waypoints = np.array([])
        self.poly_x = np.array([2.0], dtype=np.float32)
        self.poly_y = np.array([2.0], dtype=np.float32)
        self.poly_z = np.array([2.0], dtype=np.float32)
        self.max_time = 0

    def receive_waypoints_callback(self, msg):
        self.waypoints = np.array(msg.waypoints)

        self._calculate_polynomial(self.waypoints)

        pub_msg = PolynomialTrajectory()


        pub_msg.poly_x = self.poly_x.tolist()
        pub_msg.poly_y = self.poly_y.tolist()
        pub_msg.poly_z = self.poly_z.tolist()

        # pub_msg.t_clip = float(self.num_waypoints-1)
        pub_msg.t_clip = self.max_time

        self.publisher.publish(pub_msg)


    def _calculate_polynomial(self, waypoints):
        # calculate polynomial coefficients for x(t), y(t), and z(t) so they pass through the waypoints
        # waypoints = waypoints[:4]
        waypoints = np.array([*waypoints,waypoints[0]])

        self.get_logger().info(f'{waypoints}')

        n = len(waypoints)

        self.num_waypoints = n
        
        x = np.array([p.x for p in waypoints])
        y = np.array([p.y for p in waypoints])
        z = np.array([p.z for p in waypoints])
        waypoints_array : np.ndarray = np.array([x,y,z])
        # self.get_logger().info(f'{waypoints_array}')
        
        # self.get_logger().info(f'{waypoints_array.shape}')

        # make the time related to the distance between each two consequent waypoints
        t = 1*np.cumsum(np.sqrt(np.sum(np.diff(waypoints_array, axis=1)**2, axis=0)))
        # ad 0 to the begining of t
        t = np.concatenate(([0], t))
        
        # t = np.arange(n)
        
        t[-1] = t[-2] + 2
        self.max_time = float(t[-1])
        
        self.get_logger().info(f'{t}')
        # added order 
        extra = 0
        # least squares method for x(t)
        A = np.vstack([t**i for i in reversed(range(len(waypoints) + extra))]).T
        self.poly_x = np.linalg.lstsq(A, x, rcond=None)[0]
        self.get_logger().info(f'{self.poly_x=}')

        # least squares method for y(t)
        A = np.vstack([t**i for i in reversed(range(len(waypoints) + extra))]).T
        self.poly_y = np.linalg.lstsq(A, y, rcond=None)[0]

        # least squares method for z(t)
        A = np.vstack([t**i for i in reversed(range(len(waypoints) + extra))]).T
        self.poly_z = np.linalg.lstsq(A, z, rcond=None)[0]


    def calculate_polynomial(self, waypoints):
        #calculate polynomial coefficients for x(t), y(t) and z(t) so they pass through the waypoints
        self.poly_x = np.array([2.0], dtype=np.float32)
        self.poly_y = np.array([1.0], dtype=np.float32)
        self.poly_z = np.array([1.0], dtype=np.float32)




def main():
    rclpy.init()
    node = QuadrotorPolyOptimizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
