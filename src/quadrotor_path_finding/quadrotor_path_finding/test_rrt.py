import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
    def __str__(self):
        return f"({self.x}, {self.y})"

class RRT:
    def __init__(self, start, goal, occupancy_map, max_iter=1000, step_size=1, goal_sample_rate=0.1, min_dist=0.1, pause = 0.01):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.occupancy_map = occupancy_map
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.min_dist = min_dist
        self.obstacle_padding = 1
        self.nodes = [self.start]
        self.pause = pause

    def plan(self):
        plt.imshow(self.occupancy_map.T, cmap='gray')
        plt.plot(self.start.x, self.start.y, 'go', markersize=10)
        plt.plot(self.goal.x, self.goal.y, 'ro', markersize=10)
        for i in range(self.max_iter):
            if np.random.uniform() < self.goal_sample_rate:
                x, y = self.goal.x, self.goal.y
            else:
                x, y = np.random.uniform(self.occupancy_map.shape[0]), np.random.uniform(self.occupancy_map.shape[1])
            nearest_node = self.get_nearest_node(x, y)
            new_node = self.steer(nearest_node, x, y)
            if self.is_collision_free(nearest_node, new_node):
                self.nodes.append(new_node)
                new_node.parent = nearest_node
                plt.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], 'b-', linewidth=0.5)
                plt.pause(self.pause)
                if self.get_distance(new_node, self.goal) < self.min_dist:
                    self.goal.parent = new_node
                    path = self.get_path()
                    plt.plot([node[0] for node in path], [node[1] for node in path], 'r-', linewidth=1)
                    # plt.show()  # Move this line here
                    plt.pause(self.pause)
                    plt.show()
                    return path
        return None

    def get_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def get_nearest_node(self, x, y):
        distances = [self.get_distance(node, Node(x, y)) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_node, to_x, to_y):
        distance = self.get_distance(from_node, Node(to_x, to_y))
        if distance < self.step_size:
            return Node(to_x, to_y)
        else:
            theta = np.arctan2(to_y - from_node.y, to_x - from_node.x)
            x = from_node.x + self.step_size * np.cos(theta)
            y = from_node.y + self.step_size * np.sin(theta)
            return Node(x, y)

    def is_collision_free(self, node1, node2):
        # print(node1)
        # print(node2)
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        if x1 == x2 and y1 == y2:
            return True

        steps_x = np.linspace(x1, x2, 10)
        steps_y = np.linspace(y1, y2, 10)

        # print(steps_x)
        # print(steps_y)

        for i in range(len(steps_x)):
            x, y = steps_x[i], steps_y[i]
            if self.occupancy_map[int(x), int(y)]:
                return False
        if self.occupancy_map[int(node2.x), int(node2.y)]:
            return False
        return True
    def get_distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

# Example usage
occupancy_map = np.zeros((100, 100))
occupancy_map[20:80, 40:60] = 1
occupancy_map[40:60, 60:80] = 1
rrt = RRT(start=(20, 38), goal=(90, 90), occupancy_map=occupancy_map, step_size= 3,  pause = 1e-4)
path = rrt.plan()
print(path)
