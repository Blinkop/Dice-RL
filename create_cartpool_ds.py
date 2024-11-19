import gym
import numpy as np
import torch
from cartpole import RandomPolicy

from gym.envs.classic_control.cartpole import CartPoleEnv


class InfiniteCartPole(CartPoleEnv):
    def step(self, action):
        obs, reward, done, _, info = super(InfiniteCartPole, self).step(action)
        reward = -1.0 if done else 1.0
        return obs, reward, False, info, 1


# first_states, states, actions, next_states, rewards, step_num
def evaluate_policy_on_cartpole(policy: RandomPolicy, episodes=200):
    env = InfiniteCartPole()
    total_reward = 0
    total_disc_reward = 0
    ds = []

    for episode in range(episodes):
        state = env.reset()[0]
        first_state = state
        done = False
        episode_reward = 0
        episode_disc_reward = 0
        episode_len = 0

        for _ in range(250):
            old_state = state
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).to(policy._device).unsqueeze(0)
            )
            action = policy.select_action(state_tensor).item()

            state, reward, done, _, _ = env.step(action)
            episode_disc_reward += reward * (0.99**episode_len)
            episode_reward += reward
            ds.append([first_state, old_state, action, state, reward, episode_len])
            episode_len += 1

        total_reward += episode_reward
        total_disc_reward += episode_disc_reward
        # print(f"Episode {episode + 1}, reward: {episode_reward}, len: {episode_len}")
    env.close()
    average_reward = total_reward / episodes
    average_disc_reward = total_disc_reward / episodes
    print(f"Average reward over {episodes} episodes: {average_reward}")
    print(f"Average discounted reward over {episodes} episodes: {average_disc_reward}")

    return ds

if __name__ == "__main__":
    ds = evaluate_policy_on_cartpole(RandomPolicy())
    ds = np.array(ds, dtype=object)
    np.save("cartpole_ds.npy", ds, allow_pickle=True)
