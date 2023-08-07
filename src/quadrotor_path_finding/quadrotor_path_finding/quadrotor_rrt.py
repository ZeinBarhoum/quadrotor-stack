import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import OccupancyGrid3D, PathWayPoints, State
from geometry_msgs.msg import Point
import numpy as np

import matplotlib.pyplot as plt

class MapNode:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class RRT:
    def __init__(self, start, goal, occupancy_map, max_iter=1000, step_size=1, goal_sample_rate=0.1, min_dist=0.1, pause=0.01):
        self.start = MapNode(start[0], start[1], start[2])
        self.goal = MapNode(goal[0], goal[1], goal[2])
        self.occupancy_map = occupancy_map
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.min_dist = min_dist
        self.obstacle_padding = 1
        self.nodes = [self.start]
        self.pause = pause

    def plan(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(self.start.x, self.start.y, self.start.z, c='g', marker='o', s=100)
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, c='r', marker='o', s=100)


        for i in range(self.max_iter):
            if np.random.uniform() < self.goal_sample_rate:
                x, y, z = self.goal.x, self.goal.y, self.goal.z
            else:
                x, y, z = np.random.uniform(self.occupancy_map.shape[0]), np.random.uniform(self.occupancy_map.shape[1]), np.random.uniform(self.occupancy_map.shape[2])
            nearest_node = self.get_nearest_node(x, y, z)
            new_node = self.steer(nearest_node, x, y, z)
            if self.is_collision_free(nearest_node, new_node):
                self.nodes.append(new_node)
                new_node.parent = nearest_node
                ax.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], [nearest_node.z, new_node.z], 'b-', linewidth=0.5)
                plt.pause(self.pause)
                if self.get_distance(new_node, self.goal) < self.min_dist:
                    self.goal.parent = new_node
                    path = self.get_path()
                    ax.plot([node[0] for node in path], [node[1] for node in path], [node[2] for node in path], 'r-', linewidth=3)
                    # Show occupancy map
                    xs, ys, zs = np.where(self.occupancy_map > 0)
                    ax.scatter(xs, ys, zs, c='k', marker='s', s=10)

                    plt.pause(self.pause)
                    plt.show()
                    return path
        return None

    def get_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]

    def get_nearest_node(self, x, y, z):
        distances = [self.get_distance(node, MapNode(x, y, z)) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_node, to_x, to_y, to_z):
        distance = self.get_distance(from_node, MapNode(to_x, to_y, to_z))
        if distance < self.step_size:
            return MapNode(to_x, to_y, to_z)
        else:
            theta = np.arctan2(to_y - from_node.y, to_x - from_node.x)
            x = from_node.x + self.step_size * np.cos(theta)
            y = from_node.y + self.step_size * np.sin(theta)
            z = from_node.z + self.step_size * np.random.uniform(-1, 1)
            return MapNode(x, y, z)

    def is_collision_free(self, node1, node2):
        x1, y1, z1 = node1.x, node1.y, node1.z
        x2, y2, z2 = node2.x, node2.y, node2.z
        if x1 == x2 and y1 == y2 and z1 == z2:
            return True

        steps_x = np.linspace(x1, x2, 10)
        steps_y = np.linspace(y1, y2, 10)
        steps_z = np.linspace(z1, z2, 10)

        for i in range(len(steps_x)):
            x, y, z = steps_x[i], steps_y[i], steps_z[i]

            x = np.clip(x, 0, self.occupancy_map.shape[0] - 1)
            y = np.clip(y, 0, self.occupancy_map.shape[1] - 1)
            z = np.clip(y, 0, self.occupancy_map.shape[2] - 1)

            if self.occupancy_map[int(x), int(y), int(z)]:
                return False

        x, y, z = node2.x, node2.y, node2.z

        x = np.clip(x, 0, self.occupancy_map.shape[0] - 1)
        y = np.clip(y, 0, self.occupancy_map.shape[1] - 1)
        z = np.clip(y, 0, self.occupancy_map.shape[2] - 1)

        if self.occupancy_map[int(x), int(y), int(z)]:
            return False
        return True

    def get_distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)



class QuadrotorRRT(Node):
    def __init__(self):
        super().__init__('quadrotor_rrt')
        self.subscriber_map = self.create_subscription(OccupancyGrid3D, 'map', self.map_callback, 10)
        self.publisher_path = self.create_publisher(PathWayPoints, 'quadrotor_waypoints', 10)
        self.subscriber_replan = self.create_subscription(Point, 'quadrotor_plan_command', self.plan_callback, 10)
        self.subscriber_state = self.create_subscription(State, 'quadrotor_state', self.state_callback, 10)

        self.current_position = np.zeros(3)
        self.goal_position = np.ones(3)
        self.map_data = None
        self.cell_size = 1.0

    def map_callback(self, msg):
        # Store the map data in a NumPy ndarray
        self.map_data = np.array(msg.data, dtype=np.int8)
        self.map_data = self.map_data.reshape((msg.width, msg.height, msg.depth))
        self.cell_size = msg.cell_size

    def plan_callback(self, msg):
        # self.
        # Update the goal position
        self.goal_position = np.array([msg.x, msg.y, msg.z])
        self.goal_position = np.array(self.goal_position / self.cell_size).astype(np.int32)
        print(self.goal_position)

        rrt = RRT(self.current_position, goal=self.goal_position, occupancy_map=self.map_data, step_size=50, pause=1e-5, min_dist= 50)

        path = rrt.plan()
        path = np.array(path) * self.cell_size
        path = path.astype(np.float32).tolist()

        msg = PathWayPoints()
        waypoints = []
        for point in path:
            p = Point()
            p.x, p.y, p.z = point
            waypoints.append(p)
        msg.waypoints = waypoints

        # self.get_logger().info(f'{msg}')

        # Generate a new path

        # Publish the path
        self.publisher_path.publish(msg)
    def state_callback(self, msg):
        # Update the current position
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.current_position = np.array(self.current_position / self.cell_size).astype(np.int32)
        # print(self.current_position)


def main():
    rclpy.init()
    _node = QuadrotorRRT()
    rclpy.spin(_node)
    _node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
