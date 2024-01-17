""" Train RL agents of the quadrotor simulation environment """
from quadrotor_simulation.RL.envs import QuadrotorBaseEnv
# from ray.rllib.utils.spaces.space_utils import normalize_action
import torch
import stable_baselines3 as sb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import argparse
import os


def get_env(server):
    env = QuadrotorBaseEnv(wrench_actions=False, normalized_actions=True,
                           config={'physics': {'physics_server': server, 'quadrotor_description': 'neuroBEM', 'render_ground': False, 'publish_state': False}},
                           )
    return env


def main():
    parser = argparse.ArgumentParser(description='Process Learning Parameters')
    parser.add_argument('-e', dest='envs', help='Number of environment to use for collecting data', type=int, default=10)
    parser.add_argument('-m', dest='multi_processing', help='If set, use multi processing', action='store_true')
    parser.add_argument('-l', dest='model_path_load', help='Path to the model to load', type=str, default=None)
    parser.add_argument('-s', dest='model_path_save', help='Directory to save the model', type=str, default='./results/models')
    parser.add_argument('-v', dest='verbose', help='Whether to print the logs', action='store_true')
    parser.add_argument('-t', dest='tensorboard_log', help='Path to the tensorboard log', type=str, default='./results/logs/')
    parser.add_argument('-r', dest='render', help='Whether to render at least one GUI, for multi_processing this means rendering all environments', action='store_true')
    parser.add_argument('-n', dest='name', help='Name of the experiment', type=str, default='ppo_quadrotor')
    # TODO: add options for hyperparameters
    args = parser.parse_args()
    num_envs = args.envs
    vec_cls = DummyVecEnv if not args.multi_processing else SubprocVecEnv
    server = 'GUI' if args.render else 'DIRECT'
    train_env = make_vec_env(get_env, num_envs, env_kwargs={'server': server}, vec_env_cls=vec_cls)
    policy_kwargs = dict(activation_fn=lambda: torch.nn.LeakyReLU(0.01),
                         net_arch=[512, 512]
                         )
    roll_ep = 1
    ep_steps = 1000
    n_steps = roll_ep * ep_steps
    batch_size = n_steps * num_envs

    model_name = args.name
    tensorboard_log = args.tensorboard_log
    tensorboard_log = os.path.join(tensorboard_log, model_name)
    verbose = args.verbose

    model = sb.PPO("MlpPolicy", train_env, n_steps=n_steps, batch_size=batch_size, verbose=verbose,
                   policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=0.001)
    load_file = args.model_path_load
    if load_file is not None:
        model.set_parameters(load_file)
    learn_steps = int(1e8)
    try:
        model.learn(total_timesteps=learn_steps, progress_bar=True, log_interval=1)
    except KeyboardInterrupt:
        pass
    save_dir = args.model_path_save
    save_dir = os.path.join(save_dir, model_name)
    model.save(save_dir)


if __name__ == '__main__':
    main()
