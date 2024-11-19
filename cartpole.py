import abc

import gym
import torch


class Policy:
    def __init__(
        self, device: torch.device = torch.device("cuda:0"), name: str = "unnamed"
    ):
        super().__init__()

        self.policy_name = name
        self._device = device

    def _get_batch_size(self, state: torch.Tensor):
        if len(state.shape) == 1:
            return 1
        else:
            return state.shape[0]

    @abc.abstractmethod
    def select_action(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def action_dist(self, state):
        raise NotImplementedError()


class RandomPolicy(Policy):
    def __init__(
        self, device: torch.device = torch.device("cpu"), name: str = "random_policy"
    ):
        super().__init__(device, name)

    def select_action(self, state):
        """
        Select a random action (0 or 1) for CartPole.
        """
        batch_size = self._get_batch_size(state)
        return torch.randint(0, 2, (batch_size,), device=self._device)

    def action_dist(self, state):
        """
        Return a uniform distribution over actions.
        """
        batch_size = self._get_batch_size(state)
        return torch.ones((batch_size, 2), device=self._device) / 2


# Function to run the policy on CartPole environment
def evaluate_policy_on_cartpole(policy, episodes=1000):
    gamma = 0.99
    env = gym.make("CartPole-v1")
    total_reward = 0

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        episode_len = 0

        while not done:
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).to(policy._device).unsqueeze(0)
            )
            action = policy.select_action(state_tensor).item()
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward * (gamma ** episode_len)
            episode_len += 1

        total_reward += episode_reward
        print(f"Episode {episode + 1}, reward: {episode_reward}, len: {episode_len}")

    env.close()
    average_reward = total_reward / episodes
    print(f"Average reward over {episodes} episodes: {average_reward}")
    return average_reward

if __name__ == "__main__":
    # Run the random policy on CartPole
    device = torch.device("cpu")
    random_policy = RandomPolicy(device=device)
    evaluate_policy_on_cartpole(random_policy)
