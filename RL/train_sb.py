""" Train RL agents of the quadrotor simulation environment """
from quadrotor_simulation.RL.envs import QuadrotorBaseEnv
from ray.rllib.utils.spaces.space_utils import normalize_action
import torch
import stable_baselines3 as sb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def get_env():
    env = QuadrotorBaseEnv(wrench_actions=False, normalized_actions=True,
                           config={'physics': {'physics_server': 'DIRECT', 'quadrotor_description': 'neuroBEM', 'render_ground': False, 'publish_state': False}},
                           )
    return env


if __name__ == '__main__':
    num_envs = 12
    vec_cls = SubprocVecEnv
    # vec_cls = DummyVecEnv
    train_env = make_vec_env(get_env, num_envs, vec_env_cls=vec_cls)
    # for torch
    policy_kwargs = dict(activation_fn=lambda: torch.nn.LeakyReLU(0.01),
                         net_arch=[128, 128]
                         )
    roll_ep = 1
    ep_steps = 1000
    n_steps = roll_ep * ep_steps
    print(f"n_steps = {n_steps}")
    batch_size = n_steps * num_envs
    print(f"batch_size = {batch_size}")

    model = sb.PPO("MlpPolicy", train_env, n_steps=n_steps, batch_size=batch_size, verbose=1,
                   policy_kwargs=policy_kwargs, tensorboard_log="./PPO_quadrotor_tensorboard/", learning_rate=0.001)

    learn_steps = int(1e8)
    try:
        model.learn(total_timesteps=learn_steps, progress_bar=True, log_interval=1)
    except KeyboardInterrupt:
        pass
    model.save("./results/ppo_quad_imp_wrench_norm")
