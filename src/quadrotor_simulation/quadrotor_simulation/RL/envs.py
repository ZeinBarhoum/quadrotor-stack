import gymnasium as gym
import rclpy
from rclpy.node import Node
from quadrotor_interfaces.msg import RotorCommand, State
from sensor_msgs.msg import Imu, Image
from quadrotor_simulation.quadrotor_pybullet_physics import QuadrotorPybulletPhysics
from quadrotor_simulation.quadrotor_pybullet_camera import QuadrotorPybulletCamera
from quadrotor_simulation.quadrotor_imu import QuadrotorIMU
from rclpy.executors import SingleThreadedExecutor

from rclpy.parameter import Parameter
import yaml
import threading
import numpy as np
import time

from gymnasium import spaces

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec


def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    time_to_wait=-1
):
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
    wait_set.add_subscription(sub.handle)
    sigint_gc = SignalHandlerGuardCondition(context=context)
    wait_set.add_guard_condition(sigint_gc.handle)

    timeout_nsec = timeout_sec_to_nsec(time_to_wait)
    wait_set.wait(timeout_nsec)

    subs_ready = wait_set.get_ready_entities('subscription')
    guards_ready = wait_set.get_ready_entities('guard_condition')

    if guards_ready:
        if sigint_gc.handle.pointer in guards_ready:
            return (False, None)

    if subs_ready:
        if sub.handle.pointer in subs_ready:
            msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
            return (True, msg_info[0])

    return (False, None)


def run_exexutor(executor):
    executor.spin()


class QuadrotorEnv(gym.Env):
    def __init__(self, observation_type=['state'], env_suffix='', config=None):
        rclpy.init()
        self.threads = []
        if isinstance(config, dict):
            config = config
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        self._executor = SingleThreadedExecutor()

        parameters_physics = []
        parameters_physics.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
        parameters_physics.append(Parameter('use_ff_state', Parameter.Type.BOOL, True))
        if 'physics' in config:
            for key, value in config['physics'].items():
                parameters_physics.append(Parameter(name=key, value=value))

        self.physics_node = QuadrotorPybulletPhysics(suffix=env_suffix, parameter_overrides=parameters_physics)
        self._executor.add_node(self.physics_node)
        if 'image' in observation_type:
            parameters_camera = []
            parameters_camera.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
            if 'camera' in config:
                for key, value in config['camera'].items():
                    parameters_camera.append(Parameter(name=key, value=value))
            self.camera_node = QuadrotorPybulletCamera(suffix=env_suffix, parameter_overrides=parameters_camera)
            self._executor.add_node(self.camera_node)
        if 'imu' in observation_type:
            parameters_imu = []
            parameters_imu.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
            if 'imu' in config:
                for key, value in config['imu'].items():
                    parameters_imu.append(Parameter(name=key, value=value))
            self.imu_node = QuadrotorIMU(suffix=env_suffix, parameter_overrides=parameters_imu)
            self._executor.add_node(self.imu_node)
            # self.threads.append(threading.Thread(target=run_node, args=(imu_node,)))

        self.ROT_HOVER_VEL = self.physics_node.ROT_HOVER_VEL
        ROT_MAX_VEL = self.physics_node.ROT_MAX_VEL
        self.action_space = gym.spaces.Box(low=0.0, high=ROT_MAX_VEL, shape=(4,), dtype=np.float32)
        _spaces = []
        if 'state' in observation_type:
            _spaces.append(spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32))
        if 'image' in observation_type:
            _spaces.append(spaces.Box(low=0, high=255, shape=(self.camera_node.image_height, self.camera_node.image_width, 3), dtype=np.uint8))
        if 'imu' in observation_type:
            _spaces.append(spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32))
        if len(_spaces) == 1:
            self.observation_space = _spaces[0]
        else:
            self.observation_space = spaces.Tuple(_spaces)
        self.observation_type = observation_type
        self._rotor_speeds_topic = self.physics_node.rotor_speeds_topic
        self._ff_state_topic = self.physics_node.ff_state_topic
        self._state_topic = self.physics_node.state_topic
        if 'image' in observation_type:
            self._image_topic = self.camera_node.image_topic
        if 'imu' in observation_type:
            self._imu_topic = self.imu_node.imu_topic
        self._node = Node('master_node'+env_suffix)
        self._rotor_speeds_publisher = self._node.create_publisher(RotorCommand, self.physics_node.rotor_speeds_topic, 10)
        self._ff_state_publisher = self._node.create_publisher(State, self.physics_node.ff_state_topic, 10)

        self.state = self.physics_node.state
        if 'imu' in self.observation_type:
            self.imu = self.imu_node.imu
        if 'image' in self.observation_type:
            self.image = self.camera_node.ros_image

        self._thread = threading.Thread(target=run_exexutor, args=(self._executor,))
        self._thread.start()

    def reset(self):
        ff_state = State()
        ff_state.state.pose.position.z = 1.0
        self._ff_state_publisher.publish(ff_state)
        self._rotor_speeds_publisher.publish(RotorCommand(rotor_speeds=[self.ROT_HOVER_VEL]*4))

    def step(self, action):
        self._ff_state_publisher.publish(self.state)
        self._rotor_speeds_publisher.publish(RotorCommand(rotor_speeds=action))
        obs = []
        self.state = wait_for_message(State, self._node, self._state_topic)[1]
        if 'imu' in self.observation_type:
            self.imu = wait_for_message(Imu, self._node, self._imu_topic)[1]
        if 'image' in self.observation_type:
            self.image = wait_for_message(Image, self._node, self._image_topic)[1]

        if 'state' in self.observation_type:
            statedata = self.state.state
            obs.append(np.array([statedata.pose.position.x, statedata.pose.position.y, statedata.pose.position.z, statedata.pose.orientation.x, statedata.pose.orientation.y, statedata.pose.orientation.z,
                                 statedata.pose.orientation.w, statedata.twist.linear.x, statedata.twist.linear.y, statedata.twist.linear.z, statedata.twist.angular.x, statedata.twist.angular.y, statedata.twist.angular.z]))
        if 'image' in self.observation_type:
            obs.append(np.array(self.image.data).reshape((self.camera_node.image_height, self.camera_node.image_width, 3)))
        if 'imu' in self.observation_type:
            obs.append(np.array([self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.imu.angular_velocity.z,
                       self.imu.linear_acceleration.x, self.imu.linear_acceleration.y, self.imu.linear_acceleration.z]))

        if len(obs) == 1:
            obs = obs[0]
        else:
            obs = tuple(obs)
        time.sleep(0.001)
        return obs, 0, False, {}

    def close(self):
        for node in self._executor.get_nodes():
            node.destroy_node()
        self._executor.shutdown()
        self._thread.join()


env = QuadrotorEnv(env_suffix='_env1', observation_type=['state'], config={
                   'physics': {'physics_server': 'GUI', 'quadrotor_description': 'neuroBEM'}})

obs = env.reset()
for i in range(10000):
    obs, reward, done, info = env.step([env.ROT_HOVER_VEL*1.1]*4)
env.close()
