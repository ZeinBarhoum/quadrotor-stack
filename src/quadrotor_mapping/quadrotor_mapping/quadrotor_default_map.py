import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import OccupancyGrid3D
from quadrotor_utils.map_transformations import voxelmap_to_OccupancyGrid3D, point_to_voxel
from quadrotor_utils.collision_detection import *
import numpy as np


class QuadrotorDefaultMap(Node):
    def __init__(self):
        super().__init__('quadrotor_default_map')
        self.publisher_map = self.create_publisher(OccupancyGrid3D, 'quadrotor_map', 10)

        voxel_map = np.zeros((200, 200, 200))
        # occupancy_map[20:80, 40:60, 20:80] = 1
        # occupancy_map[40:60, 60:80, 20:80] = 1
        # occupancy_map[:, :, 0:2] = 1

        self.voxel_map = voxel_map

        self.grid = voxelmap_to_OccupancyGrid3D(self.voxel_map, 0.1)

        # Create a timer to publish the map
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):

        self.publisher_map.publish(self.grid)


def main():
    rclpy.init()
    node = QuadrotorDefaultMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
