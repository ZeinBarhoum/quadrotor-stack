#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ardrone_interfaces.msg import NAVDataDemo
from ardrone_interfaces.msg import NAVDataEulerAngles
from ardrone_interfaces.msg import NAVDataMagneto
from ardrone_interfaces.msg import NAVDataPressureRaw
from ardrone_interfaces.msg import NAVDataPwm
from ardrone_interfaces.msg import NAVDataRawMeasures
from ardrone_interfaces.msg import NAVDataTime
from ardrone_interfaces.srv import Command

# import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2

import logging
import time
import pyardrone

from pyardrone import ARDrone
from pyardrone.navdata import options as nav
from pyardrone import at
import threading


class DroneNode(Node):
    """A ROS2 Node that connects and controls the AR.Drone 2.0"""

    is_shutdown = False             # use for stoping theards
    navdata_bit_config = 0          # bit view of configuration for getting navdata
    command_is_executing = True    # use for stoping theards

    def __init__(self):
        super().__init__("ardrone_driver_node", parameter_overrides=[])
        self.get_logger().info("AR.Drone driver node has been started.")
        logging.basicConfig(level=logging.INFO)
        self.droneLog = logging.getLogger("AR.Drone")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('timeout', 0.03),
                ('watchdog_interval', 0.03),
                ('navdata_options', "{'demo': True}")
            ])

        self.timeout = self.get_parameter('timeout').value
        self.watchdog_interval = self.get_parameter('watchdog_interval').value
        self.navdata_options = eval(self.get_parameter('navdata_options').value)

        self.init_drone()

        self.command_is_executing = False

        self.service_server = self.create_service(
            srv_type=Command,
            srv_name='/ardrone_control_command',
            callback=self.command_callback)

        self.subscription_control_cmd_twist = self.create_subscription(
            Twist,
            'ardrone_control_cmd_twist',
            self.control_cmd_twist_listener,
            10)
        self.subscription_control_cmd_twist
        self.subscription_control_cmd_pwm = self.create_subscription(
            Int32MultiArray,
            'ardrone_control_cmd_pwm',
            self.control_cmd_pwm_listener,
            10)
        self.subscription_control_cmd_pwm

        self.data_thread = threading.Thread(name='updateData', target=self.data_callback)
        self.data_thread.start()
        self.video_thread = threading.Thread(name='updateData', target=self.video_callback)
        self.video_thread.start()

    def NAVDataOptionsConfig(self, options):
        """Configurate NAV data getting from AR.Drone 2.0"""

        for option, flag in options.items():                    # check input dictionary for matches in navdata options
            if option == 'demo' and flag:
                continue
            is_absent = True
            for index_register, class_type in nav.index.items():
                if class_type._attrname == option:
                    # if class_type.__name__ == option:
                    is_absent = False
                    if flag:                                    # update navdata configuration
                        self.navdata_bit_config = self.navdata_bit_config | (1 << index_register)
                    else:
                        self.navdata_bit_config = self.navdata_bit_config & ~(1 << index_register)
                    break
            if is_absent:
                self.droneLog.info("Don't find option: " + option + "!")

        if self.navdata_bit_config != 0 and \
                'demo' not in options.keys() or options['demo'] == False:       # if configuration is not empty, but Demo is False
            options['demo'] = True
            self.droneLog.info("You need Demo for getting data! Added Demo to NAV data.")

        self.navdata_options = options

        self.droneLog.info("NAV data: " + str(self.navdata_options))

        self.droneLog.info("Demo configurating.")
        self.drone.send(pyardrone.at.CONFIG('general:navdata_demo', options['demo']))
        self.configWait()

        self.droneLog.info("NAV Data options configurating.")
        self.droneLog.info("Configurating message: " + bin(self.navdata_bit_config))
        self.drone.send(pyardrone.at.CONFIG('general:navdata_options', self.navdata_bit_config))
        self.configWait()

        return True, ''

    def command_callback(self,
                         request: Command.Request,
                         response: Command.Response
                         ) -> Command.Response:
        response.success = False
        response.message = ''
        try:
            if self.command_is_executing:
                response.message = 'Drone is executing command'
            else:
                # self.droneLog.info(' Get command: ' + request.command)
                func = getattr(self, request.command)
                response.success, response.message = func()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as err:
            response.success = False
            response.message = str(err)
        return response

    def control_cmd_twist_listener(self, cmd_twist):
        if self.drone.state.fly_mask and not self.command_is_executing:
            self.move_drone_twist(pitch_velocity=cmd_twist.linear.x,
                                  roll_velocity=cmd_twist.linear.y,
                                  gaz_velocity=cmd_twist.linear.z,
                                  yaw_velocity=cmd_twist.angular.z)

    def control_cmd_pwm_listener(self, cmd_array_pwm):
        if not self.command_is_executing:
            self.move_drone_pwm(motor1=cmd_array_pwm.data[0],
                                motor2=cmd_array_pwm.data[1],
                                motor3=cmd_array_pwm.data[2],
                                motor4=cmd_array_pwm.data[3])

    def video_callback(self):
        self.video_publisher = self.create_publisher(Image, 'ardrone_video_frames', 10)
        self.br = CvBridge()        # Used to convert between ROS and OpenCV images
        while not (self.is_shutdown or self.drone.closed):
            # timer_start = self.get_clock().now().nanoseconds
            try:
                ret, frame = self.get_frame()
                # self.get_logger().info(f"{ret}")
                if ret == True:
                    # cv2.imshow('frame', frame)
                    # cv2.waitKey(1)
                    msg = self.br.cv2_to_imgmsg(frame, encoding='rgb8')
                    msg.header.stamp = self.get_clock().now().to_msg()
                    self.video_publisher.publish(msg)
            except Exception as err:
                if err == KeyboardInterrupt:
                    raise KeyboardInterrupt
                # self.droneLog.info("Video ERROR: " + str(err))
            time.sleep(self.timeout)

    def data_callback(self):
        # timer_start = self.get_clock().now().nanoseconds
        # timer_timeout = self.watchdog_interval * 1000000

        if 'demo' in self.navdata_options:
            self.navdata_Demo_publisher = self.create_publisher(NAVDataDemo, 'ardrone_navdata_Demo', 10)
        if 'euler_angles' in self.navdata_options:
            self.navdata_EulerAngles_publisher = self.create_publisher(NAVDataEulerAngles, 'ardrone_navdata_EulerAngles', 10)
        if 'magneto' in self.navdata_options:
            self.navdata_Magneto_publisher = self.create_publisher(NAVDataMagneto, 'ardrone_navdata_Magneto', 10)
        if 'pressure_raw' in self.navdata_options:
            self.navdata_PressureRaw_publisher = self.create_publisher(NAVDataPressureRaw, 'ardrone_navdata_PressureRaw', 10)
        if 'pwm' in self.navdata_options:
            self.navdata_Pwm_publisher = self.create_publisher(NAVDataPwm, 'ardrone_navdata_Pwm', 10)
        if 'raw_measures' in self.navdata_options:
            self.navdata_RawMeasures_publisher = self.create_publisher(NAVDataRawMeasures, 'ardrone_navdata_RawMeasures', 10)
        if 'time' in self.navdata_options:
            self.navdata_Time_publisher = self.create_publisher(NAVDataTime, 'ardrone_navdata_Time', 10)

        while not (self.is_shutdown or self.drone.closed):

            try:
                ret, navdata = self.get_navdata()
                if ret == True:
                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    if 'demo' in self.navdata_options:
                        header.frame_id = "Demo"
                        self.navdata_Demo_publish(header, navdata.demo)
                    if 'euler_angles' in self.navdata_options:
                        header.frame_id = "EulerAngles"
                        self.navdata_EulerAngles_publish(header, navdata.euler_angles)
                    if 'magneto' in self.navdata_options:
                        header.frame_id = "Magneto"
                        self.navdata_Magneto_publish(header, navdata.magneto)
                    if 'pressure_raw' in self.navdata_options:
                        header.frame_id = "PressureRaw"
                        self.navdata_PressureRaw_publish(header, navdata.pressure_raw)
                    if 'pwm' in self.navdata_options:
                        header.frame_id = "Pwm"
                        self.navdata_Pwm_publish(header, navdata.pwm)
                    if 'raw_measures' in self.navdata_options:
                        header.frame_id = "RawMeasures"
                        self.navdata_RawMeasures_publish(header, navdata.raw_measures)
                    if 'time' in self.navdata_options:
                        header.frame_id = "Time"
                        self.navdata_Time_publish(header, navdata.time)
            except Exception as err:
                if err == KeyboardInterrupt:
                    raise KeyboardInterrupt
                self.droneLog.info("NAV data ERROR: " + str(err))

            # timer_wait = (self.get_clock().now().nanoseconds - timer_start - timer_timeout) / 1000000000.0
            # if (timer_wait > 0):
            #     time.sleep(timer_wait)
            time.sleep(self.timeout)

    def navdata_Demo_publish(self, header, data_demo):
        msg = NAVDataDemo()
        msg.header = header
        msg.orientation = Vector3()
        msg.twist_linear = Vector3()
        # Flying state (landed, flying, hovering, etc.)
        msg.ctrl_state = data_demo.ctrl_state
        # battery voltage filtered (mV)
        msg.vbat_flying_percentage = data_demo.vbat_flying_percentage
        # UAV's pitch in milli-degrees
        msg.orientation.x = data_demo.phi
        # UAV's roll  in milli-degrees
        msg.orientation.y = -data_demo.theta
        # UAV's yaw   in milli-degrees
        msg.orientation.z = -data_demo.psi
        # UAV's estimated linear velocity
        msg.twist_linear.x = data_demo.vx
        # UAV's estimated linear velocity
        msg.twist_linear.y = data_demo.vy
        # UAV's estimated linear velocity
        msg.twist_linear.z = data_demo.vz
        # UAV's altitude in centimeters
        msg.altitude = data_demo.altitude
        self.navdata_Demo_publisher.publish(msg)

    def navdata_EulerAngles_publish(self, header, data_euler_angles):
        msg = NAVDataEulerAngles()
        msg.header = header
        msg.theta = data_euler_angles.theta_a
        msg.phi = data_euler_angles.phi_a
        self.navdata_EulerAngles_publisher.publish(msg)

    def navdata_Magneto_publish(self, header, data_magneto):
        msg = NAVDataMagneto()
        msg.header = header
        msg.magneto_raw = Vector3()
        msg.magneto_rectified = Vector3()
        msg.magneto_offset = Vector3()
        msg.mx = data_magneto.mx
        msg.my = data_magneto.my
        msg.mz = data_magneto.mz
        # magneto in the body frame, in mG
        msg.magneto_raw.x = float(data_magneto.magneto_raw[0])
        msg.magneto_raw.y = float(data_magneto.magneto_raw[1])
        msg.magneto_raw.z = float(data_magneto.magneto_raw[2])
        msg.magneto_rectified.x = float(data_magneto.magneto_rectified[0])
        msg.magneto_rectified.y = float(data_magneto.magneto_rectified[1])
        msg.magneto_rectified.z = float(data_magneto.magneto_rectified[2])
        msg.magneto_offset.x = float(data_magneto.magneto_offset[0])
        msg.magneto_offset.y = float(data_magneto.magneto_offset[1])
        msg.magneto_offset.z = float(data_magneto.magneto_offset[2])
        self.navdata_Magneto_publisher.publish(msg)

    def navdata_PressureRaw_publish(self, header, data_pressure_raw):
        msg = NAVDataPressureRaw()
        msg.header = header
        msg.up = data_pressure_raw.up
        msg.ut = data_pressure_raw.ut
        msg.temperature_meas = data_pressure_raw.Temperature_meas
        msg.pression_meas = data_pressure_raw.Pression_meas
        self.navdata_PressureRaw_publisher.publish(msg)

    def navdata_Pwm_publish(self, header, data_pwm):
        msg = NAVDataPwm()
        msg.header = header
        msg.motor = [data_pwm.motor1,
                     data_pwm.motor2,
                     data_pwm.motor3,
                     data_pwm.motor4]
        msg.sat_motor = [data_pwm.sat_motor1,
                         data_pwm.sat_motor2,
                         data_pwm.sat_motor3,
                         data_pwm.sat_motor4]
        msg.gaz_feed_forward = data_pwm.gaz_feed_forward
        msg.gaz_altitude = data_pwm.gaz_altitude
        msg.altitude_integral = data_pwm.altitude_integral
        msg.vz_ref = data_pwm.vz_ref
        msg.u_angle = [data_pwm.u_pitch,
                       data_pwm.u_roll,
                       data_pwm.u_yaw]
        msg.yaw_u_i = data_pwm.yaw_u_I
        msg.u_angle_planif = [data_pwm.u_pitch_planif,
                              data_pwm.u_roll_planif,
                              data_pwm.u_yaw_planif]
        msg.u_gaz_planif = data_pwm.u_gaz_planif
        msg.current_motor = [data_pwm.current_motor1,
                             data_pwm.current_motor2,
                             data_pwm.current_motor3,
                             data_pwm.current_motor4]
        msg.altitude_prop = data_pwm.altitude_prop
        msg.altitude_der = data_pwm.altitude_der
        self.navdata_Pwm_publisher.publish(msg)

    def navdata_RawMeasures_publish(self, header, data_raw_measures):
        msg = NAVDataRawMeasures()
        msg.header = header
        msg.raw_imu = Imu()
        # A covariance matrix of all zeros will be interpreted as "covariance unknown", and to use the data a covariance will have to be assumed or gotten from some other source
        msg.raw_imu.orientation_covariance[0] = -1
        # filtered accelerometers
        msg.raw_imu.linear_acceleration.x = float(data_raw_measures.raw_accs[0])
        msg.raw_imu.linear_acceleration.y = float(data_raw_measures.raw_accs[1])
        msg.raw_imu.linear_acceleration.z = float(data_raw_measures.raw_accs[2])
        # filtered gyrometers
        msg.raw_imu.angular_velocity.x = float(data_raw_measures.raw_gyros[0])
        msg.raw_imu.angular_velocity.y = float(data_raw_measures.raw_gyros[1])
        msg.raw_imu.angular_velocity.z = float(data_raw_measures.raw_gyros[2])
        # gyrometers  x/y 110 deg/s
        msg.raw_gyros_110 = [int(data_raw_measures.raw_gyros_110[0]), int(data_raw_measures.raw_gyros_110[1])]
        # battery voltage raw (mV)
        msg.raw_vbat = data_raw_measures.vbat_raw
        msg.us_debut_echo = data_raw_measures.us_debut_echo
        msg.us_fin_echo = data_raw_measures.us_fin_echo
        msg.us_association_echo = data_raw_measures.us_association_echo
        msg.us_distance_echo = data_raw_measures.us_distance_echo
        msg.us_courbe_temps = data_raw_measures.us_courbe_temps
        msg.us_courbe_valeur = data_raw_measures.us_courbe_valeur
        msg.us_courbe_ref = data_raw_measures.us_courbe_ref
        msg.flag_echo_ini = data_raw_measures.flag_echo_ini
        msg.nb_echo = data_raw_measures.nb_echo
        msg.sum_echo = data_raw_measures.sum_echo
        msg.alt_temp_raw = data_raw_measures.alt_temp_raw
        msg.gradient = data_raw_measures.gradient
        self.navdata_RawMeasures_publisher.publish(msg)

    def navdata_Time_publish(self, header, data_time):
        msg = NAVDataTime()
        msg.header = header
        msg.header.frame_id = "Time"
        # 32 bit value where the 11 most significant bits represents the seconds,
        # and the 21 least significant bits are the microseconds.
        msg.time = data_time.time
        self.navdata_Time_publisher.publish(msg)

    def init_drone(self):
        self.droneLog.info("Initializing ARDrone")

        self.droneLog.info("Setuping ARDrone.")
        self.drone = ARDrone(timeout=self.timeout, watchdog_interval=self.watchdog_interval)
        self.recrash()
        self.drone.navdata_ready.wait()
        self.droneLog.info("NavData ready.")
        # self.drone.video_ready.wait()
        # self.droneLog.info("Video ready")

        self.droneLog.info("Video channel configurating. Select front camera.")
        self.drone.send(pyardrone.at.CONFIG("video:video_channel", "0"))
        self.configWait()
        self.isFrontCamera = True

        self.NAVDataOptionsConfig(self.navdata_options)

        self.droneLog.info("Finish initializing ARDrone.")

        return True, ''

    def closeDrone(self):
        self.is_shutdown = True
        self.drone.close()

    # Drone accept the config

    def configWait(self, timeout=5):
        self.droneLog.info("Wait for configurating")
        time.sleep(timeout)
        self.droneLog.info("Configurating is finish")

    def take_off(self):
        self.droneLog.info("Taking off!")
        self.command_is_executing = True

        while not self.drone.state.fly_mask:
            self.drone.takeoff()

        # self.video_reconnect()

        self.drone.video_ready.wait()
        self.droneLog.info("Video ready")

        self.command_is_executing = False
        self.droneLog.info("Taked off")

        return True, ''

    def land(self):
        if not self.drone.state.fly_mask:
            self.droneLog.info("Drone on ground!")
            return False, 'Drone on ground!'

        self.command_is_executing = True

        self.droneLog.info("Landing!")
        while self.drone.state.fly_mask:
            self.drone.land()

        # self.video_reconnect()

        self.command_is_executing = False
        self.droneLog.info("Landed")

        return True, 'Landed'

    def emergency_land(self):
        if not self.drone.state.fly_mask:
            self.droneLog.info("Drone on ground!")
            return False, 'Drone on ground!'

        self.droneLog.info("Emergency landing!")
        while self.drone.state.fly_mask:
            self.drone.emergency()
        self.droneLog.info("Emergency landed.")

        return True, 'Emergency landed'

    def move_drone_pwm(self, motor1=0, motor2=0, motor3=0, motor4=0):
        if motor1 >= 0 and motor1 <= 511 and\
           motor2 >= 0 and motor2 <= 511 and\
           motor3 >= 0 and motor3 <= 511 and\
           motor4 >= 0 and motor4 <= 511:
            self.drone.send(at.PWM(motor1, motor2, motor3, motor4))
            return True, ''
        else:
            return False, 'Input pwm value error'

    # move_code:
    # 1 - forward
    # 2 - backward
    # 3 - left
    # 4 - right
    # 5 - up
    # 6 - down
    # 7 - cw
    # 8 - ccw
    def moveDrone(self, velocity, move_code):
        if not self.command_is_executing and self.drone.state.fly_mask:
            if move_code == 1:
                self.drone.move(forward=velocity)

            elif move_code == 2:
                self.drone.move(backward=velocity)

            elif move_code == 3:
                self.drone.move(left=velocity)

            elif move_code == 4:
                self.drone.move(right=velocity)

            elif move_code == 5:
                self.drone.move(up=velocity)

            elif move_code == 6:
                self.drone.move(down=velocity)

            elif move_code == 7:
                self.drone.move(cw=velocity)

            elif move_code == 8:
                self.drone.move(ccw=velocity)

    # Velocity in free space broken into its linear and angular parts.
    # pitch_velocity - linear X
    # roll_velocity  - linear Y
    # gaz_velocity   - linear Z
    # yaw_velocity   - angular Z
    def move_drone_twist(self, pitch_velocity=0, roll_velocity=0, gaz_velocity=0, yaw_velocity=0):
        if not self.command_is_executing and self.drone.state.fly_mask:
            pitch_velocity *= -1
            yaw_velocity *= -1

            # PCMD(flag=<flag.progressive: 1>, roll=0, pitch=0.8, gaz=0, yaw=0)
            self.drone.send(at.PCMD(at.PCMD.flag.progressive, roll_velocity, pitch_velocity, gaz_velocity, yaw_velocity))
            # self.drone.move(roll=roll_velocity, pitch=pitch_velocity, gaz=gaz_velocity, yaw=yaw_velocity)
            return True, ''

        return False, 'Drone on ground!'

    # switch between cameras

    def switch_camera(self):
        if self.isFrontCamera:
            self.isFrontCamera = False
            self.drone.send(pyardrone.at.CONFIG("video:video_channel", "1"))
            self.droneLog.info("Switched to bottom camera.")
        else:
            self.isFrontCamera = True
            self.drone.send(pyardrone.at.CONFIG("video:video_channel", "0"))
            self.droneLog.info("Switched to front camera.")
        self.configWait(timeout=0)

        return True, ''

    # use this comand after drone crash to tell him that its ok now

    def recrash(self):
        self.get_logger().info("Trimming")
        self.drone.emergency()
        self.drone.ftrim()
        self.get_logger().info("Trimmed")
        self.drone.mtrim(0, 0, 0.5)
        self.get_logger().info("Trimmed")

        return True, 'Ready to fly!'

    def get_frame(self):
        if not self.command_is_executing and self.drone.video_ready:
            return True, self.drone.frame

        return False, None

    def get_navdata(self):
        if not self.command_is_executing and self.drone.navdata_ready:
            return True, self.drone.navdata

        return False, None

    def video_reconnect(self):
        self.drone.reconnect_VideoMixin()


def main(args=None):
    rclpy.init(args=args)
    drone_node = DroneNode()
    try:
        rclpy.spin(drone_node)
    except KeyboardInterrupt:
        print("SHUT DOWN")
    except Exception as e:
        print(e)
    drone_node.closeDrone()
    rclpy.shutdown()
    # try:
    #     rclpy.init(args=args)

    #     drone_node = DroneNode()

    #     rclpy.spin(drone_node)
    # except KeyboardInterrupt:
    #     print("SHUT DOWN")
    #     pass
    # except Exception as e:
    #     print(e)
    # finally:
    #     try:
    #         drone_node.closeDrone()
    #     except:
    #         pass
    #     rclpy.shutdown()


if __name__ == "__main__":
    main()
