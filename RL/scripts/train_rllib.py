""" Train RL agents of the quadrotor simulation environment """
from quadrotor_simulation.RL.envs import QuadrotorBaseEnv
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env


def env_creator(_):
    return QuadrotorBaseEnv()  # return an environment instance


register_env("QuadrotorBaseEnv", env_creator)  # register the environment

if __name__ == '__main__':
    num_envs = 1
    config = {
        "env": "QuadrotorBaseEnv",
        "num_workers": num_envs,
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
        },
        "lr": 0.001,
        "rollout_fragment_length": 1000,
        "train_batch_size": 1000 * num_envs,
        "sgd_minibatch_size": 1000 * num_envs,
        "num_sgd_iter": 1,
    }

    stop = {
        "training_iteration": int(1e8),
    }

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment(QuadrotorBaseEnv)
    )

    algo = config.build()  # 2. build the algorithm,
