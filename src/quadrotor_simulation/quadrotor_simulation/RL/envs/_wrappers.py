from numpy import shape
from ._base import QuadrotorBaseEnv
import gymnasium as gym
import numpy as np


class QuadNormAction(QuadrotorBaseEnv):
    """ A wrapper around QuadrotorBaseEnv for normalized actions The actions are normalized to be in the range [0,1]. The maximum rotor_speed is retreived from env.MAX_ROT_VEL
    The actions mapping works as follows:
    action = rotor_speed /ROT_MAX_VEL
    """

    def __init__(self, observation_type=..., env_suffix='', config=None, time_limit=-1, terminate_on_contact=False):
        super().__init__(observation_type, env_suffix, config, time_limit, terminate_on_contact)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def step(self, action):
        rotor_speeds = np.array(action) * self.ROT_MAX_VEL
        return super().step(rotor_speeds)
