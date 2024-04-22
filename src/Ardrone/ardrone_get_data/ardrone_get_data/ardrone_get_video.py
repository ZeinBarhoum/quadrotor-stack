import rclpy 
from rclpy.node import Node 

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import cv2
 

class VideoSubscriber(Node):

    def __init__(self):
        super().__init__('ardrone_video_subscriber')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # from the video_frames topic. The queue size is 10 messages.
        self.video_subscription = self.create_subscription(
            Image,
            'ardrone_video_frames',
            self.listener_video_callback,
            10)
        self.video_subscription # prevent unused variable warning

        # self.navdata_subscription = self.create_subscription(
        #     String,
        #     'ardrone_navdata',
        #     self.listener_navdata_callback,
        #     10)
        # self.navdata_subscription # prevent unused variable warning


    def listener_video_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        cv2.imshow("AR.Drone camera", current_frame)     # Display image
        cv2.waitKey(1)


    def listener_navdata_callback(self, data):
        print(data.data)
        print("_______")
        

def main(args=None):
    try:
        rclpy.init(args=args)

        video_subscriber = VideoSubscriber()

        rclpy.spin(video_subscriber)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        video_subscriber.destroy_node()
        rclpy.shutdown()
   

if __name__ == '__main__':
    main()