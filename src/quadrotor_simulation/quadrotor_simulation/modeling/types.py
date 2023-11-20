from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm


@dataclass
class QuadState:
    """State is a pose and twist of the quadrotor. Could be represented in any frame of reference."""
    _pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # xyzw
    _vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        self._pos = np.array(new_pos).reshape(3)

    @property
    def quat(self):
        return self._quat

    @quat.setter
    def quat(self, new_quat):
        self._quat = np.array(new_quat).reshape(4)

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, new_vel):
        self._vel = np.array(new_vel).reshape(3)

    @property
    def ang_vel(self):
        return self._ang_vel

    @ang_vel.setter
    def ang_vel(self, new_ang_vel):
        self._ang_vel = np.array(new_ang_vel).reshape(3)

    def as_vector(self):
        return np.hstack([self.pos, self.quat, self.vel, self.ang_vel]).reshape(-1, 1)

    def fill_state_msg(self, msg):
        msg.state.pose.position.x = self.pos[0]
        msg.state.pose.position.y = self.pos[1]
        msg.state.pose.position.z = self.pos[2]
        msg.state.pose.orientation.x = self.quat[0]
        msg.state.pose.orientation.y = self.quat[1]
        msg.state.pose.orientation.z = self.quat[2]
        msg.state.pose.orientation.w = self.quat[3]
        msg.state.twist.linear.x = self.vel[0]
        msg.state.twist.linear.y = self.vel[1]
        msg.state.twist.linear.z = self.vel[2]
        msg.state.twist.angular.x = self.ang_vel[0]
        msg.state.twist.angular.y = self.ang_vel[1]
        msg.state.twist.angular.z = self.ang_vel[2]
        return msg


@dataclass
class QuadInput:
    """Input is an angular velocity and angular acceleration of the quadrotor's rotors."""
    _ang_vels: np.ndarray = field(default_factory=lambda: np.zeros(4))
    _ang_accs: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def ang_vels(self):
        return self._ang_vels

    @ang_vels.setter
    def ang_vels(self, new_ang_vels):
        self._ang_vels = np.array(new_ang_vels).reshape(4)

    @property
    def ang_accs(self):
        return self._ang_accs

    @ang_accs.setter
    def ang_accs(self, new_ang_accs):
        self._ang_accs = np.array(new_ang_accs).reshape(4)

    def as_vector(self):
        return np.hstack([self.ang_vels, self.ang_accs]).reshape(-1, 1)

    def fill_rotor_msg(self, msg):
        msg.rotor_speeds = self.ang_vels


@dataclass
class QuadAccel:
    """Accel is a linear and angular acceleration of the quadrotor. Could be represented in any frame of reference."""
    _acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _ang_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, new_acc):
        self._acc = np.array(new_acc).reshape(3)

    @property
    def ang_acc(self):
        return self._ang_acc

    @ang_acc.setter
    def ang_acc(self, new_ang_acc):
        self._ang_acc = np.array(new_ang_acc).reshape(3)

    def __add__(self, other):
        return QuadAccel(self.acc + other.acc, self.ang_acc + other.ang_acc)

    def __sub__(self, other):
        return QuadAccel(self.acc - other.acc, self.ang_acc - other.ang_acc)

    def as_vector(self):
        return np.hstack([self.acc, self.ang_acc]).reshape(-1, 1)

    def fill_state_msg(self, msg):
        msg.state.accel.linear.x = self.acc[0]
        msg.state.accel.linear.y = self.acc[1]
        msg.state.accel.linear.z = self.acc[2]
        msg.state.accel.angular.x = self.ang_acc[0]
        msg.state.accel.angular.y = self.ang_acc[1]
        msg.state.accel.angular.z = self.ang_acc[2]
        return msg


@dataclass
class QuadWrench:
    """Wrench is a force and torque applied to the quadrotor. Could be represented in any frame of reference."""
    _force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _torque: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, new_force):
        self._force = np.array(new_force).reshape(3)

    @property
    def torque(self):
        return self._torque

    @torque.setter
    def torque(self, new_torque):
        self._torque = np.array(new_torque).reshape(3)

    def __add__(self, other):
        return QuadWrench(self.force + other.force, self.torque + other.torque)

    def __sub__(self, other):
        return QuadWrench(self.force - other.force, self.torque - other.torque)

    def as_vector(self):
        return np.hstack([self.force, self.torque]).reshape(-1, 1)


def skew_symmetric(vec):
    """Returns skew-symmetric matrix of a vector"""
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def euler_integration_step(state: QuadState, acc: QuadAccel, params):
    """Performs a single step of euler integration

    Args:
        state (QuadState): current state of the quadrotor (pose + twist)
        acc (QuadAccel): current acceleration of the quadrotor
        params (dict): dictionary including the key 'dt' with the time step

    Returns:
        QuadState: new state of the quadrotor after dt
    """
    dt = params['dt']
    new_state = QuadState()
    new_state.pos = state.pos + state.vel * dt
    new_state.vel = state.vel + acc.acc * dt

    rotm = R.from_quat(state.quat).as_matrix()
    new_rotm = rotm @ expm(skew_symmetric(state.ang_vel * dt))
    new_state.quat = R.from_matrix(new_rotm).as_quat()

    new_state.ang_vel = state.ang_vel + acc.ang_acc * dt
    return new_state
