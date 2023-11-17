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

import cv2
import pybullet as p
from tqdm import tqdm


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

        parameters_physics = []
        parameters_physics.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
        parameters_physics.append(Parameter('use_ff_state', Parameter.Type.BOOL, True))
        if 'physics' in config:
            for key, value in config['physics'].items():
                parameters_physics.append(Parameter(name=key, value=value))

        self.physics_node = QuadrotorPybulletPhysics(suffix=env_suffix, parameter_overrides=parameters_physics)
        if 'image' in observation_type:
            parameters_camera = []
            parameters_camera.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
            if 'camera' in config:
                for key, value in config['camera'].items():
                    parameters_camera.append(Parameter(name=key, value=value))
            self.camera_node = QuadrotorPybulletCamera(suffix=env_suffix, parameter_overrides=parameters_camera)
        if 'imu' in observation_type:
            parameters_imu = []
            parameters_imu.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
            if 'imu' in config:
                for key, value in config['imu'].items():
                    parameters_imu.append(Parameter(name=key, value=value))
            self.imu_node = QuadrotorIMU(suffix=env_suffix, parameter_overrides=parameters_imu)

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

        self.state = self.physics_node.state
        if 'imu' in self.observation_type:
            self.imu = self.imu_node.imu
        if 'image' in self.observation_type:
            self.image = self.camera_node.ros_image

    def reset(self):
        ff_state = State()
        ff_state.state.pose.position.z = 1.0
        self.physics_node.receive_ff_state_callback(ff_state)
        self.physics_node.receive_commands_callback(RotorCommand(rotor_speeds=[self.ROT_HOVER_VEL]*4))

    def step(self, action):
        self.physics_node.receive_commands_callback(RotorCommand(rotor_speeds=action))
        obs = []
        self.state = self.physics_node.state
        if 'imu' in self.observation_type:
            self.imu_node.receive_state_callback(self.state)
            self.imu = self.imu_node.imu
        if 'image' in self.observation_type:
            self.camera_node.receive_state_callback(self.state)
            self.image = self.camera_node.ros_image

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
        return obs, 0, False, {}

    def close(self):
        self.physics_node.destroy_node()
        rclpy.shutdown()


env = QuadrotorEnv(env_suffix='_env1', observation_type=['state', 'image', 'imu'], config={
                   'physics': {'physics_server': 'DIRECT', 'quadrotor_description': 'neuroBEM', 'render_ground': True, 'publish_state': False},
                   'camera': {'image_width': 640, 'image_height': 480, 'camera_fov': 60.0, 'physics_server': 'GUI', 'quadrotor_description': 'neuroBEM'}})

obs = env.reset()
action = env.action_space.sample()
for i in tqdm(range(10000)):
    obs, reward, done, info = env.step(action)
    cv2.imshow('', obs[1])
    cv2.waitKey(1)
env.close()
