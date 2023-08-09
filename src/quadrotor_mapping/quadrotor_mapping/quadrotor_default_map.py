import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import OccupancyGrid3D
import numpy as np


class QuadrotorDefaultMap(Node):
    def __init__(self):
        super().__init__('quadrotor_default_map')
        self.publisher_map = self.create_publisher(OccupancyGrid3D, 'map', 10)

        occupancy_map = np.zeros((100, 100, 100))
        occupancy_map[20:80, 40:60, 20:80] = 1
        occupancy_map[40:60, 60:80, 20:80] = 1
        occupancy_map[:,:,0:2] = 1

        self.map = occupancy_map

        self.map_data = self.map_to_occupancygrid(self.map)


        # Create a timer to publish the map
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):

        self.publisher_map.publish(self.map_data)


    def map_to_occupancygrid(self, map: np.ndarray):
        # Convert the NumPy array to a ROS message
        map_data = OccupancyGrid3D()

        map_data.width = map.shape[0]
        map_data.height = map.shape[1]
        map_data.depth = map.shape[2]

        map_data.cell_size = 0.05

        map_data.data = map.flatten().astype(np.int8).tolist()

        return map_data

def main():
    rclpy.init()
    node = QuadrotorDefaultMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()