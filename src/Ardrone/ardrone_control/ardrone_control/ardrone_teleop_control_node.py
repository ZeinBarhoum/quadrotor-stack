import sys
import threading

import pty
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.task import Future

import geometry_msgs.msg
from ardrone_interfaces.srv import Command
from std_msgs.msg import Empty
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

import include.logitech_joistick as joystick


class ArdroneTeleopControlNode(Node):
    """A ROS2 Node with a Service Server for WhatIsThePoint."""
    instruction_keyboard_msg = \
        """
    --Comands--
    arrow_up      : take off
    arrow_down    : land
    p : emergency land (turn off motors)
    o : recrash (allow drone to fly after crash)
    z : switch camera
    Anything else : stop moving

    --Linear moving--
    w : forwards (+x)
    s : backwards (-x)
    d : right (+y)
    a : left (-y)
    r : up (+z)
    f : down (-z)

    --Angular moving--
    e : right (+z)
    q : left (-z)
    """

    instruction_joystick_msg = \
        """
    --Comands--
    buttons on joystick handle on the right : take off / land
    joystick handle trigger                 : recrash (allow drone to fly after crash)
    button on joystick handle under thumb   : switch camera

    --Linear/Angular moving--
    slider                                  : limits maximum speed
    buttons on joystick handle on the left  : up (+z) / down (-z) (use max availible speed)
    joystick handle control                 : remaining control (lin x/y, ang z)

    !!!To start moving you need at least once move joystick handle and slider!!!
    """

    moveBindings = {
        'w': (1, 0, 0, 0),
        's': (-1, 0, 0, 0),
        'd': (0, 1, 0, 0),
        'a': (0, -1, 0, 0),
        'r': (0, 0, 1, 0),
        'f': (0, 0, -1, 0),
        'q': (0, 0, 0, 1),
        'e': (0, 0, 0, -1),
    }

    def __init__(self):
        super().__init__("ardrone_teleop_control_node")
        self.get_logger().info("AR.Drone teleop control node has been started.")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('work_mode', 0),
                ('use_joystick', True),
                ('max_lin_speed', 0.3),
                ('max_ang_speed', 1.0)
            ])

        self.work_mode = self.get_parameter('work_mode').value
        self.use_joystick = self.get_parameter('use_joystick').value
        self.max_lin_speed = self.get_parameter('max_lin_speed').value
        self.max_ang_speed = self.get_parameter('max_ang_speed').value

        self.get_logger().info(f"{self.work_mode}, {self.use_joystick}, {self.max_lin_speed}, {self.max_ang_speed}")

        self.settings = self.saveTerminalSettings()

        self.service_client = self.create_client(
            srv_type=Command,
            srv_name='/ardrone_control_command')

        if (self.work_mode == 0):
            TwistMsg = geometry_msgs.msg.Twist
            self.twist_msg = TwistMsg()
            self.cmd_twist_publisher = self.create_publisher(TwistMsg, 'ardrone_control_cmd_twist', 10)
            print('MODE 0 : Twist control')
            if self.use_joystick:
                print(self.instruction_joystick_msg)
            else:
                print(self.instruction_keyboard_msg)
        elif (self.work_mode == 1):
            self.pwm_msg = Int32MultiArray()
            self.motors_power = 0
            self.cmd_pwm_publisher = self.create_publisher(Int32MultiArray, 'ardrone_control_cmd_pwm', 10)
            print('MODE 1 : PWM control')

        self.future: Future = None
        self.joy = joystick.LogitechJoistick()
        self.checkedJoy = False

    def send_cmd_command(self, cmd_msg):
        request = Command.Request()
        request.command = cmd_msg

        if self.future is not None and not self.future.done():
            self.future.cancel()  # Cancel the future. The callback will be called with Future.result == None.
            self.get_logger().info("Service Future canceled. The Node took too long to process the service call.")
        self.future = self.service_client.call_async(request)
        self.future.add_done_callback(self.process_response)

    def process_response(self, future: Future):
        """Callback for the future, that will be called when it is done"""
        response = future.result()

        success = response.success
        message = response.message

        if response is not None:
            self.get_logger().info(('Success.' if success else 'Fail!') + ((" Callback message: " + message) if message != '' else ''))
        else:
            self.get_logger().info("The response was None.")

    def getKey(self, settings):
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def saveTerminalSettings(self):
        return termios.tcgetattr(sys.stdin.fileno())

    def restoreTerminalSettings(self, old_settings):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def updateJoy(self):
        try:
            self.joy.update()
            return True
        except Exception as exp:
            self.use_joystick = False
            print("Joystick not accepted!")
            print(str(exp))
            print("Keyboard start.")
            print(self.instruction_keyboard_msg)
            return False

    def joystick_publish_command(self):
        if self.joy.cmd_takeoff:
            self.joy.cmd_takeoff = 0
            self.send_cmd_command('take_off')
        elif self.joy.cmd_land:
            self.joy.cmd_land = 0
            self.send_cmd_command('land')
        elif self.joy.cmd_switchcamera:
            self.joy.cmd_switchcamera = 0
            self.send_cmd_command('switch_camera')
        elif self.joy.cmd_recrash:
            self.joy.cmd_recrash = 0
            self.send_cmd_command('recrash')
        elif self.joy.cmd_video_reconnect:
            self.joy.cmd_video_reconnect = 0
            self.send_cmd_command('video_reconnect')
        else:
            if self.checkedJoy:
                if self.work_mode == 0:
                    self.twist_msg.linear.x = self.joy.val_lin_twistX * self.joy.val_max_speed
                    self.twist_msg.linear.y = self.joy.val_lin_twistY * self.joy.val_max_speed
                    self.twist_msg.linear.z = self.joy.val_lin_twistZ * self.joy.val_max_speed
                    self.twist_msg.angular.x = 0.0
                    self.twist_msg.angular.y = 0.0
                    self.twist_msg.angular.z = self.joy.val_ang_twistZ * self.max_ang_speed
                    self.cmd_twist_publisher.publish(self.twist_msg)
                elif self.work_mode == 1:
                    power = int(self.joy.val_max_speed*511)
                    self.pwm_msg.data = [power, power, power, power]
                    self.cmd_pwm_publisher.publish(self.pwm_msg)
            else:
                self.checkedJoy = True
                for check in self.joy.permit_check:
                    if not check:
                        self.checkedJoy = False
                if self.checkedJoy:
                    print('Joystic twist control availible now.')

    def keyboard_publish_command(self):
        key = self.getKey(self.settings)
        if key == '\x1b':
            # Если escape последовательность, то считать еще 2 символа
            # Но будет некорректно работать, если был нажата клавиша Escape (будет ждать нажатия еще 2 кнопок)
            key += sys.stdin.read(2)
        # print(repr(key))
        if key == '\x1b[A':                 # Arrow Up. Take off drone.
            self.send_cmd_command('take_off')
        elif key == '\x1b[B':               # Arrow Down. Land drone.
            self.send_cmd_command('land')
        elif key == 'p':                    # Turn off motors.
            self.send_cmd_command('emergency_land')
        elif key == 'o':                    # Allow drone to fly after crash.
            self.send_cmd_command('recrash')
        elif key == 'z':                    # Switch camera.
            self.send_cmd_command('switch_camera')
        elif key in self.moveBindings.keys():
            if self.work_mode == 0:
                self.twist_msg.linear.x = self.moveBindings[key][0] * self.max_lin_speed
                self.twist_msg.linear.y = self.moveBindings[key][1] * self.max_lin_speed
                self.twist_msg.linear.z = self.moveBindings[key][2] * self.max_lin_speed
                self.twist_msg.angular.x = 0.0
                self.twist_msg.angular.y = 0.0
                self.twist_msg.angular.z = self.moveBindings[key][3] * self.max_ang_speed
                self.cmd_twist_publisher.publish(self.twist_msg)
            elif self.work_mode == 1:
                self.motors_power = int(self.motors_power + self.moveBindings[key][0] * 10)
                if self.motors_power < 0:
                    self.motors_power = 0
                elif self.motors_power > 511:
                    self.motors_power = 511
                self.pwm_msg.data = [self.motors_power, self.motors_power, self.motors_power, self.motors_power]
                self.cmd_pwm_publisher.publish(self.pwm_msg)
        else:
            if (key == '\x03'):
                raise KeyboardInterrupt


def main(args=None):
    try:
        rclpy.init(args=args)

        control_node = ArdroneTeleopControlNode()

        # rclpy.spin(control_node)
        spinner = threading.Thread(target=rclpy.spin, args=(control_node,))
        spinner.start()

        try:
            while True:
                if (control_node.use_joystick):
                    if not control_node.updateJoy():
                        continue
                    control_node.joystick_publish_command()
                else:
                    control_node.keyboard_publish_command()
        except KeyboardInterrupt:
            pass

        if control_node.work_mode == 0:
            control_node.twist_msg.linear.x = 0.0
            control_node.twist_msg.linear.y = 0.0
            control_node.twist_msg.linear.z = 0.0
            control_node.twist_msg.angular.x = 0.0
            control_node.twist_msg.angular.y = 0.0
            control_node.twist_msg.angular.z = 0.0
            control_node.cmd_twist_publisher.publish(control_node.twist_msg)
            control_node.send_cmd_command('land')
        elif control_node.work_mode == 1:
            pwm_array = [int(0), int(0), int(0), int(0)]
            control_node.cmd_pwm_publisher.publish(Int32MultiArray(data=pwm_array))
        control_node.restoreTerminalSettings(control_node.settings)

    except KeyboardInterrupt:
        pass
    # except Exception as e:
    #     print(e)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
