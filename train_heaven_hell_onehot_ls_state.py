import numpy as np
import gym
env = gym.make('gym_custom:heaven-hell-onehot-ls-v0')

from replay_buffer import ReplayBuffer
from params_pool import ParamsPool

buf = ReplayBuffer(size=10000)
param = ParamsPool(
    input_dim=20,  # number of states
    num_actions=env.action_space.n,  # number of actions
    epsilon_multiplier=0.99,
    epsilon_min=0.01
)
target_network_update_duration = 10
max_steps = 40
batch_size = 32

num_episodes = 1000

for e in range(num_episodes):

    _ = env.reset()

    total_reward = 0
    total_steps = 0

    while True:

        # ===== getting the tuple (s, a, r, s', a') =====

        state = env._one_hot(env.get_state(), 20)
        action = param.act_using_behavior_policy(state)

        _, reward, done, _ = env.step(action)

        # logistics
        total_reward += reward
        total_steps += 1
        if total_steps >= max_steps: done = True

        next_state = env._one_hot(env.get_state(), 20)
        next_action = param.act_using_target_policy(next_state)

        mask = 0 if done else 1

        # ===== storing it to the buffer =====

        buf.push(state, action, reward, next_state, next_action, mask)

        # ===== update the parameters =====

        param.update_q_prediction_net(buf.sample(batch_size=batch_size))

        # ===== check done =====

        if done: break

    # ===== after an episode =====

    if e % target_network_update_duration == 0:
        param.update_q_target_net()

    param.decay_epsilon()

    print(f'Episode {e:3.0f} | Return {total_reward:3.0f} | Episode Len {total_steps:3} | Epsilon {param.epsilon:3.2f}')

env.close()