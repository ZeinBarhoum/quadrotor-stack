from geometry_msgs.msg import Wrench
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from quadrotor_interfaces.msg import RotorCommand, State
import rclpy
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
import yaml
from scipy.optimize import lsq_linear

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
            # client = p.connect(p.GUI)
            # p.disconnect(client)
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
        self.MAX_THRUST = self.physics_node.MAX_THRUST
        self.MAX_TORQUEX = self.physics_node.MAX_TORQUEX
        self.MAX_TORQUEY = self.physics_node.MAX_TORQUEY
        self.MAX_TORQUEZ = self.physics_node.MAX_TORQUEZ
        self.J = self.physics_node.J
        self.normalized_actions = normalized_actions
        self.wrench_actions = wrench_actions
        if not self.wrench_actions:
            if self.normalized_actions:
                self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=0.0, high=self.ROT_MAX_VEL, shape=(4,), dtype=np.float32)
        else:
            if not self.normalized_actions:
                self.action_space = gym.spaces.Box(low=np.array([-self.MAX_THRUST, -self.MAX_TORQUEX, -self.MAX_TORQUEY, -self.MAX_TORQUEZ]),
                                                   high=np.array([self.MAX_THRUST, self.MAX_TORQUEX, self.MAX_TORQUEY, self.MAX_TORQUEZ]),
                                                   dtype=np.float32)
            else:
                # MASS NORMALIZED
                self.action_space = gym.spaces.Box(low=np.array([-self.MAX_THRUST, -self.MAX_TORQUEX, -self.MAX_TORQUEY, -self.MAX_TORQUEZ])/self.M,
                                                   high=np.array([self.MAX_THRUST, self.MAX_TORQUEX, self.MAX_TORQUEY, self.MAX_TORQUEZ])/self.M,
                                                   dtype=np.float32)

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

        self.workspace = np.array([[-2, 2], [-2, 2], [0, 2]])
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
            obs, _, _, _, info = self.step(np.array([0.0, 0.0, 0.0, 0.0]))
        else:
            obs, _, _, _, info = self.step(hover_action)
        self.time = 0
        self.closed = False
        info['t'] = self.time
        return obs, info

    def step(self, action):
        if self.closed:
            raise Exception("Trying to step in closed environment")
        self.time += self.dt
        action = np.array(action, dtype=float)
        if self.wrench_actions:
            if self.normalized_actions:
                action = action * self.M
            msg = Wrench()
            msg.force.x, msg.force.y, msg.force.z = 0.0, 0.0, action[0]
            msg.torque.x, msg.torque.y, msg.torque.z = action[1:]
            self.physics_node.receive_wrench_command_callback(msg)
        else:
            if self.normalized_actions:
                action = action * self.ROT_MAX_VEL
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
        reached, stabilized = self.check_reached_stabilized()
        if stabilized:
            terminated = True
            info['is_success'] = True

        reward = self.get_reward(obs, terminated, truncated, reached, stabilized)

        return obs, reward, terminated, truncated, info

    def check_reached_stabilized(self):
        pos = np.array([self.state.state.pose.position.x, self.state.state.pose.position.y, self.state.state.pose.position.z])
        goal = np.array(self.goal)
        dist = np.linalg.norm(pos-goal)
        vel = np.array([self.state.state.twist.linear.x, self.state.state.twist.linear.y, self.state.state.twist.linear.z])
        vel_norm = np.linalg.norm(vel)
        avel = np.array([self.state.state.twist.angular.x, self.state.state.twist.angular.y, self.state.state.twist.angular.z])
        avel_norm = np.linalg.norm(avel)
        stabilized = False
        reached = False
        if dist < 0.1:
            reached = True
        if dist < 0.1 and vel_norm < 0.1 and avel_norm < 0.1:
            stabilized = True
        return reached, stabilized

    def get_reward(self, obs, terminated, truncated, reached, stabilized):
        pos = obs[:3]
        goal = np.array(self.goal)
        dist = np.linalg.norm(pos-goal)
        dist_reward = 1. / (1 + dist**2)
        quats = obs[3:7]
        z_axis = Rotation.from_quat(quats).apply(np.array([0, 0, 1]))
        tiltage = np.abs(1 - z_axis[2])
        tilt_reward = 1. / (1 + tiltage**2)
        ang_vel_z = obs[-1]
        spinnage = np.abs(ang_vel_z)
        spinnage_reward = 1. / (1+spinnage**2)
        # reward = -dist - terminated*100
        # print(f"dist: {dist}, tiltage: {tiltage}, spinnage: {spinnage}")
        # print(f"dist_reward: {dist_reward}, tilt_reward: {tilt_reward}, spinnage_reward: {spinnage_reward}")
        reward = dist_reward + dist_reward * (tilt_reward + spinnage_reward)
        if reached:
            reward += 10
        if terminated:
            if stabilized:
                reward += 1000
            else:
                reward -= 100
        return reward

    def dfbc_agent_action(self):
        reference_position = np.array([0, 0, 1])
        actual_position = np.array([self.state.state.pose.position.x, self.state.state.pose.position.y, self.state.state.pose.position.z])
        reference_velocity = np.array([0, 0, 0])
        actual_velocity = np.array([self.state.state.twist.linear.x, self.state.state.twist.linear.y, self.state.state.twist.linear.z])
        reference_orientation = np.array([0, 0, 0, 1])
        actual_orientation = np.array([self.state.state.pose.orientation.x, self.state.state.pose.orientation.y,
                                      self.state.state.pose.orientation.z, self.state.state.pose.orientation.w])
        reference_orientation_euler = Rotation.from_quat(reference_orientation).as_euler('xyz', degrees=False)
        actual_orientation_euler = Rotation.from_quat(actual_orientation).as_euler('xyz', degrees=False)
        reference_angular_velocity = np.array([0, 0, 0])
        actual_angular_velocity = np.array([self.state.state.twist.angular.x, self.state.state.twist.angular.y, self.state.state.twist.angular.z])
        reference_linear_acceleration = np.array([0, 0, 0])
        KP_XYZ = np.array([10.0, 10.0, 10.0])  # For position
        KD_XYZ = np.array([6.0, 6.0, 6.0])  # For position
        KP_RPY = np.array([150, 150, 10.0])  # For roll, pitch and yaw
        KD_RPY = np.array([20, 20, 8.0])  # For roll, pitch and yaw
        Weights = np.array([0.001, 10, 10, 0.1])

        error_position = reference_position - actual_position
        error_velocity = reference_velocity - actual_velocity
        error_orientation = reference_orientation_euler - actual_orientation_euler
        error_angular_velocity = reference_angular_velocity - actual_angular_velocity

        desired_acceleration = reference_linear_acceleration + np.multiply(KP_XYZ, error_position) + np.multiply(KD_XYZ, error_velocity)
        desired_acceleration[2] += self.physics_node.G
        desired_force = self.M * desired_acceleration
        desired_thrust = np.dot(desired_force, Rotation.from_quat(actual_orientation).as_matrix()[:, 2])
        # desired_thrust = np.clip(desired_thrust, 0.0, self.MAX_THRUST)

        desired_zb = desired_force / np.linalg.norm(desired_force)
        desired_xc = np.array([math.cos(reference_orientation_euler[2]), math.sin(reference_orientation_euler[2]), 0])
        desired_yb = np.cross(desired_zb, desired_xc) / np.linalg.norm(np.cross(desired_zb, desired_xc))
        desired_xb = np.cross(desired_yb, desired_zb)

        desired_Rb = np.vstack([desired_xb, desired_yb, desired_zb]).transpose()
        # desired_Rb = Rotation.from_euler('xyz', [0, 20, 0], degrees=True).as_matrix()
        actual_Rb = Rotation.from_quat(actual_orientation).as_matrix()

        error_rotation = -0.5*(desired_Rb.transpose() @ actual_Rb - actual_Rb.transpose() @ desired_Rb)
        error_rotation = np.array([error_rotation[2, 1], error_rotation[0, 2], error_rotation[1, 0]])
        # error_rotation[0] = np.clip(error_rotation[0], -1, 1)
        # error_rotation[1] = np.clip(error_rotation[1], -1, 1)
        # self.get_logger().info(f'{error_rotation=}')
        # self.get_logger().info(f'{Rotation.from_matrix(actual_Rb).as_euler("xyz", degrees=True)}')

        error_angular_velocity = reference_angular_velocity - actual_angular_velocity

        desired_ang_acceleration = np.multiply(KP_RPY, error_rotation) + np.multiply(KD_RPY, error_angular_velocity)
        desired_torques = self.J @ desired_ang_acceleration + np.cross(actual_angular_velocity, self.J @ actual_angular_velocity)
        if self.wrench_actions:
            action = np.array([desired_thrust, desired_torques[0], desired_torques[1], desired_torques[2]])
            if self.normalized_actions:
                action /= self.M
            return action
        else:
            thrust = desired_thrust
            torques = desired_torques
            self.KF = self.physics_node.KF
            self.KM = self.physics_node.KM
            self.ARM_X = self.physics_node.ARM_X
            self.ARM_Y = self.physics_node.ARM_Y
            self.ROTOR_DIRS = self.physics_node.ROTOR_DIRS
            self.MAX_RPM = self.physics_node.ROT_MAX_VEL
            A = np.array([[self.KF, self.KF, self.KF, self.KF],
                          self.KF*self.ARM_Y*np.array([-1, 1, 1, -1]),
                          self.KF*self.ARM_X*np.array([-1, -1, 1, 1]),
                          [-self.ROTOR_DIRS[0]*self.KM, -self.ROTOR_DIRS[1]*self.KM, -self.ROTOR_DIRS[2]*self.KM, -self.ROTOR_DIRS[3]*self.KM]])
            # rotor_speeds_squared = np.matmul(np.linalg.inv(A), np.array([thrust, torques[0], torques[1], torques[2]]))
            # rotor_speeds_squared = np.clip(rotor_speeds_squared, 0, self.MAX_RPM**2)
            W = np.diag(np.sqrt(Weights))
            # self.get_logger().info(f'{W=}')
            rotor_speeds_squared = lsq_linear(W@A, (W@np.array([thrust, torques[0], torques[1], torques[2]]
                                                               ).reshape(-1, 1)).flatten(), bounds=(0, self.MAX_RPM**2)).x
            # self.get_logger().info(f"{rotor_speeds_squared}")
            rotor_speeds = np.sqrt(rotor_speeds_squared)
            # actual_thrust = self.KF * np.sum(rotor_speeds_squared)
            # actual_torques = np.array([self.ARM * self.KF * (rotor_speeds_squared[0] - rotor_speeds_squared[2]),
            #                            self.ARM * self.KF * (rotor_speeds_squared[1] - rotor_speeds_squared[3]),
            #                            self.KM * (rotor_speeds_squared[0] - rotor_speeds_squared[1] + rotor_speeds_squared[2] - rotor_speeds_squared[3])])
            rotor_speeds = rotor_speeds.astype(np.float32)

            if self.normalized_actions:
                rotor_speeds /= self.MAX_RPM
            return rotor_speeds

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
