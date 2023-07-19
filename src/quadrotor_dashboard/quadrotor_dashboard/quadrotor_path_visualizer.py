import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import PathWayPoints, State
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

class QuadrotorPathVisualizer(Node):
    def __init__(self):
        super().__init__('quadrotor_path_visualizer')
        self.subscriber_waypoints = self.create_subscription(msg_type= PathWayPoints,
                                                             topic= 'quadrotor_waypoints',
                                                             callback= self.waypoints_callback,
                                                             qos_profile= 10
                                                             )
        # self.subscriber_reference = self.create_subscription(msg_type= State,
        #                                                      tooic= 'quadrotor_reference',
        #                                                      callback= self.reference_callback,
        #                                                      qos_profile= 10)

        self.subscrber_referece_path = self.create_subscription(msg_type= State,
                                                                topic = 'quadrotor_reference',
                                                                callback= self.reference_callback,
                                                                qos_profile= 10)
        
        self.timer_plot = self.create_timer(timer_period_sec= 0.1, 
                                            callback= self.plot_callback)
        # self.fig = plt.figure()
        # self.ax3 = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        
        self.fig, self.ax3 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        self.references = [[0],[0],[0]]
        self.current_reference = [0,0,0]

    def waypoints_callback(self, msg):
        # waypoints = np.array([[p.x, p.y, p.z] for p in msg.waypoints])
        # self.ax3.scatter3D(waypoints[:,1], waypoints[:,2], waypoints[:,3])
        # print(msg)
        pass
    def plot_callback(self):
        
        self.ax3.plot(*self.references, c='r')
        self.ax3.scatter(*self.current_reference, c='b', marker='o')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def reference_callback(self, msg):

        self.references[0].append(msg.pose.position.x)
        self.references[1].append(msg.pose.position.y)
        self.references[2].append(msg.pose.position.z)
        
        self.current_reference = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        
        # self.ax3.scatter(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        # self.ax3.scatter(*self.references, c='r', marker='o')
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # self.ax3.scatter(*self.references, c='r', marker='o')
        # # self.ax3.scatter(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        # plt.pause(0.01)
        # plt.show()


def main():
    rclpy.init()
    node = QuadrotorPathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__name__':
    main()



