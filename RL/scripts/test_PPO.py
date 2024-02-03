
from quadrotor_simulation.RL.envs import QuadrotorBaseEnv
import time
import numpy as np
import stable_baselines3 as sb

# env1 = QuadrotorEnvBase(env_suffix='1', config={
#                         'physics': {'physics_server': 'GUI'}})


# env = QuadrotorBaseEnv(env_suffix='',
#                        observation_type=['state'],
#                        terminate_on_contact=False,
#                        time_limit=20,
#                        normalized_actions=True,
#                        wrench_actions=True,
#                        config={'physics': {'physics_server': 'GUI', 'quadrotor_description': 'neuroBEM', 'render_ground': False, 'publish_state': False}})
ep_steps = 1000
dt = 1/240
ep_time = ep_steps * dt
ep_steps = int(int(ep_time)/dt)

server = 'GUI'

env = QuadrotorBaseEnv(env_suffix='',
                       observation_type=['state'],
                       terminate_on_contact=False,
                       time_limit=int(ep_time),
                       config={'physics': {'physics_server': server, 'quadrotor_description': 'neuroBEM', 'render_ground': False, 'publish_state': False}},
                       normalized_actions=True,
                       wrench_actions=False,
                       )


# model = sb.PPO.load("ppo_quad_imp_wrench_norm", env=env, verbose=1, tensorboard_log="./PPO_quadrotor_tensorboard/")
model = sb.PPO.load("RL/results/models/ppo_rotors_norm_no_bonus")

num_exp = 10
total_rewards = []
for _ in range(num_exp):
    total_reward = 0
    obs, info = env.reset()
    while True:
        # print(env.dfbc_agent_action())
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        # obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        time.sleep(1/240)
        if terminated or truncated:
            break
    total_rewards.append(total_reward)

    print(total_reward)

print(f"On average, this agent gets {np.mean(total_rewards)} reward per episode")
