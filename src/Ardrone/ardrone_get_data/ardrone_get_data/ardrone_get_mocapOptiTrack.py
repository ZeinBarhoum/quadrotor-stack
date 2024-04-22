import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped

from mocap_msgs.msg import Marker
from mocap_msgs.msg import Markers
from mocap_msgs.msg import RigidBody
from mocap_msgs.msg import RigidBodies


class MocapOptiTrackSubscriber(Node):

    def __init__(self):
        super().__init__('ardrone_video_subscriber')

        self.mocap_subscription = self.create_subscription(
            RigidBodies,
            'rigid_bodies',
            self.ardroneRigidBody_listener,
            10)
        self.mocap_subscription

        self.rigidbody_publisher = self.create_publisher(PoseStamped, 'ardrone_RigidBodyOptiTrack', 10)

    def ardroneRigidBody_listener(self, rigidBodies):
        try:
            rb = PoseStamped()
            firstBody = RigidBody()
            firstBody = rigidBodies.rigidbodies[0]
            rb.header = rigidBodies.header
            rb.pose = firstBody.pose
            rb2 = rb
            rb2.pose.position.x = -rb.pose.position.y
            rb2.pose.position.y = rb.pose.position.x
            rb2.pose.position.z = rb.pose.position.z
            rb2.pose.orientation.x = rb.pose.orientation.x
            rb2.pose.orientation.y = -rb.pose.orientation.y
            rb2.pose.orientation.z = rb.pose.orientation.z
            rb2.pose.orientation.w = rb.pose.orientation.w

            self.rigidbody_publisher.publish(rb)
        except Exception as err:
            if err == KeyboardInterrupt:
                raise KeyboardInterrupt
            print(err)


def main(args=None):
    try:
        rclpy.init(args=args)

        mocap_subscriber = MocapOptiTrackSubscriber()

        rclpy.spin(mocap_subscriber)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        mocap_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
