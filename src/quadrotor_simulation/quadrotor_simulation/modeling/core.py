from dataclasses import dataclass, field
import numpy as np


@dataclass
class State:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # xyzw
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # define adding and subtracting states
    def __add__(self, other):
        return State(self.pos + other.pos, self.quat + other.quat, self.vel + other.vel, self.ang_vel + other.ang_vel)

    def __sub__(self, other):
        return State(self.pos - other.pos, self.quat - other.quat, self.vel - other.vel, self.ang_vel - other.ang_vel)

    def as_vector(self):
        return np.hstack([self.pos, self.quat, self.vel, self.ang_vel]).reshape(-1, 1)


@dataclass
class Accelerations:
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __add__(self, other):
        return Accelerations(self.acc + other.acc, self.ang_acc + other.ang_acc)

    def __sub__(self, other):
        return Accelerations(self.acc - other.acc, self.ang_acc - other.ang_acc)

    def as_vector(self):
        return np.hstack([self.acc, self.ang_acc]).reshape(-1, 1)


@dataclass
class Rotors:
    mot: np.ndarray = field(default_factory=lambda: np.zeros(4))
    dmot: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def as_vector(self):
        return np.hstack([self.mot, self.dmot]).reshape(-1, 1)


@dataclass
class ExternalEffects:
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __add__(self, other):
        return ExternalEffects(self.force + other.force, self.torque + other.torque)

    def __sub__(self, other):
        return ExternalEffects(self.force - other.force, self.torque - other.torque)

    def as_vector(self):
        return np.hstack([self.force, self.torque]).reshape(-1, 1)
