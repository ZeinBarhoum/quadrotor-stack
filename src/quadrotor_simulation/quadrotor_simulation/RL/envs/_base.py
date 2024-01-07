from geometry_msgs.msg import Wrench
import gymnasium as gym
import pybullet as p
from gymnasium import spaces
import numpy as np
from quadrotor_interfaces.msg import RotorCommand, State
import rclpy
from rclpy.parameter import Parameter
import yaml

from quadrotor_simulation.quadrotor_imu import QuadrotorIMU
from quadrotor_simulation.quadrotor_pybullet_camera import QuadrotorPybulletCamera
from quadrotor_simulation.quadrotor_pybullet_physics import QuadrotorPybulletPhysics


class QuadrotorBaseEnv(gym.Env):
    """
    Base class for quadrotor environments
    The environment uses the QuadrotorPybulletPhysics node to simulate the quadrotor. However, instead of using ROS for communication, the node's callbacks are called directly.
    In addition to the physics node, the environment can also use the QuadrotorPybulletCamera and QuadrotorIMU nodes to simulate the camera and IMU sensors.
    The environment can be configured using a config file or a dictionary. 
    The configs are directly mapped to the parameters of the nodes.
    """

    def __init__(self, observation_type=['state'], env_suffix='', config=None, time_limit=-1, terminate_on_contact=False, normalized_actions=False, wrench_actions=False):
        try:
            rclpy.init()
        except Exception as e:
            print(f"Could not initialize ROS2 got error {e}")
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
        parameters_physics.append(Parameter('publish_state', Parameter.Type.BOOL, False))
        if 'physics' in config:
            for key, value in config['physics'].items():
                parameters_physics.append(Parameter(name=key, value=value))
        try:
            # check if there's a possiblity of creating a GUI environment
            client = p.connect(p.GUI)
            p.disconnect(client)
            self.physics_node = QuadrotorPybulletPhysics(suffix=env_suffix, parameter_overrides=parameters_physics)
        except Exception as e:
            if config['physics']['physics_server'] == 'GUI':
                print("Could not initialize physics node in GUI mode, trying without GUI")
                parameters_physics.append(Parameter('physics_server', Parameter.Type.STRING, 'DIRECT'))
                self.physics_node = QuadrotorPybulletPhysics(suffix=env_suffix, parameter_overrides=parameters_physics)
            else:
                raise e
        if 'image' in observation_type:
            parameters_camera = []
            parameters_camera.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
            parameters_camera.append(Parameter('publish_image', Parameter.Type.BOOL, False))
            if 'camera' in config:
                for key, value in config['camera'].items():
                    parameters_camera.append(Parameter(name=key, value=value))
            self.camera_node = QuadrotorPybulletCamera(suffix=env_suffix, parameter_overrides=parameters_camera)
        if 'imu' in observation_type:
            parameters_imu = []
            parameters_imu.append(Parameter('sequential_mode', Parameter.Type.BOOL, True))
            parameters_imu.append(Parameter('publish_imu', Parameter.Type.BOOL, False))
            if 'imu' in config:
                for key, value in config['imu'].items():
                    parameters_imu.append(Parameter(name=key, value=value))
            self.imu_node = QuadrotorIMU(suffix=env_suffix, parameter_overrides=parameters_imu)

        self.ROT_HOVER_VEL = self.physics_node.ROT_HOVER_VEL
        self.ROT_MAX_VEL = self.physics_node.ROT_MAX_VEL
        self.M = self.physics_node.M
        self.W = self.physics_node.W
        self.T2W = self.physics_node.T2W
        self.MAX_THRUST = self.W * self.T2W
        self.MAX_ONE_THRUST = self.MAX_THRUST/4
        self.J = self.physics_node.J
        self.normalized_actions = normalized_actions
        self.wrench_actions = wrench_actions
        if not self.wrench_actions:
            if self.normalized_actions:
                self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=0.0, high=self.ROT_MAX_VEL, shape=(4,), dtype=np.float32)
        else:
            pass

        _spaces = []
        if 'state' in observation_type:
            _spaces.append(spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32))
        if 'image' in observation_type:
            _spaces.append(spaces.Box(low=0, high=255, shape=(self.camera_node.image_height, self.camera_node.image_width, 3), dtype=np.uint8))
        if 'imu' in observation_type:
            _spaces.append(spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32))
        if len(_spaces) == 1:
            self.observation_space = _spaces[0]
        else:
            self.observation_space = spaces.Tuple(_spaces)
        self.observation_type = observation_type

        self.state = self.physics_node.state
        if 'imu' in self.observation_type:
            self.imu = self.imu_node.imu
        if 'image' in self.observation_type:
            self.image = self.camera_node.ros_image

        self.time = 0
        self.time_limit = time_limit
        self.terminate_on_contact = terminate_on_contact

        self.workspace = np.array([[-10, 10], [-10, 10], [0, 10]])
        self.goal = [0, 0, 1]

        self.closed = False
        self.dt = self.physics_node.simulation_step_period

    def reset(self, *, seed=None, options=None):
        if self.closed:
            raise Exception("Trying to reset a closed environment")
        np.random.seed(seed)
        ff_state = State()
        # ff_state.state.pose.position.z = 1.0
        ff_state.state.pose.position.x = np.random.uniform(self.workspace[0][0], self.workspace[0][1])
        ff_state.state.pose.position.y = np.random.uniform(self.workspace[1][0], self.workspace[1][1])
        ff_state.state.pose.position.z = np.random.uniform(self.workspace[2][0], self.workspace[2][1])
        self.physics_node.receive_ff_state_callback(ff_state)
        hover_action = [self.ROT_HOVER_VEL]*4
        if self.normalized_actions:
            hover_action /= self.ROT_MAX_VEL
        if self.wrench_actions:
            obs, _, _, _, info = self.step(np.array([0, 0, 0, 0]))
        else:
            obs, _, _, _, info = self.step(hover_action)
        self.time = 0
        self.closed = False
        return obs, info

    def step(self, action):
        if self.closed:
            raise Exception("Trying to step in closed environment")
        self.time += self.dt
        action = np.array(action)
        if self.normalized_actions:
            action = action * self.ROT_MAX_VEL
        if self.wrench_actions:
            msg = Wrench()
            msg.force.x, msg.force.y, msg.force.z = 0.0, 0.0, action[0]
            msg.torque.x, msg.torque.y, msg.torque.z = action[1:]
            self.physics_node.receive_wrench_command_callback(msg)
        else:
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
            obs.append(np.array([statedata.pose.position.x, statedata.pose.position.y, statedata.pose.position.z,
                                 statedata.pose.orientation.x, statedata.pose.orientation.y, statedata.pose.orientation.z, statedata.pose.orientation.w,
                                 statedata.twist.linear.x, statedata.twist.linear.y, statedata.twist.linear.z,
                                 statedata.twist.angular.x, statedata.twist.angular.y, statedata.twist.angular.z]))
        if 'image' in self.observation_type:
            obs.append(np.array(self.image.data).reshape((self.camera_node.image_height, self.camera_node.image_width, 3)))
        if 'imu' in self.observation_type:
            obs.append(np.array([self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.imu.angular_velocity.z,
                       self.imu.linear_acceleration.x, self.imu.linear_acceleration.y, self.imu.linear_acceleration.z]))

        if len(obs) == 1:
            obs = obs[0]
        else:
            obs = tuple(obs)
        info = {'t': self.time}
        contacted = self.physics_node.check_contact()
        if contacted:
            info['contact'] = True
        truncated = False
        if self.time > self.time_limit and self.time_limit > 0:
            truncated = True
        terminated = False
        if self.terminate_on_contact and contacted:
            terminated = True
        reward = self.get_reward(obs, terminated, truncated)

        return obs, reward, terminated, truncated, info

    def get_reward(self, obs, terminated, truncated):
        pos = obs[:3]
        goal = np.array(self.goal)
        dist = np.linalg.norm(pos-goal)
        reward = -dist - terminated*100
        return reward

    def close(self):
        print("Closing")
        if not self.closed:
            self.physics_node.destroy_node()
            if 'image' in self.observation_type:
                self.camera_node.destroy_node()
            if 'imu' in self.observation_type:
                self.imu_node.destroy_node()
            self.closed = True

    def __del__(self):
        self.close()
