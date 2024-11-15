import abc
from typing import List

import torch

class Policy(object):
    def __init__(
        self,
        device: torch.device = torch.device('cuda:0'),
        name: str = 'unnamed'
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


class AbstractRandomPolicy(Policy):
    def __init__(
        self,
        device = torch.device('cuda:0'),
        name = 'unnamed',
        seed: int = None
    ):
        super().__init__(device, name)

        self._torch_generator = torch.Generator(device=self._device)
        if seed is not None:
            self._torch_generator.manual_seed(seed)
        else:
            self._torch_generator.seed()


class RandomPolicy(AbstractRandomPolicy):
    def __init__(
        self,
        num_actions: int,
        device = torch.device('cuda:0'),
        name = 'unnamed',
        seed: int = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self._num_actions = num_actions

    def select_action(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        return torch.randint(
            0,
            self._num_actions,
            size=(batch_size,),
            generator=self._torch_generator
        ).to(self._device)
    
    def action_dist(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        action_dist = (1 / self._num_actions) * torch.ones((batch_size, self._num_actions))

        return action_dist.to(self._device)


class PopularRandomPolicy(AbstractRandomPolicy):
    def __init__(
        self,
        items_count: List[int],
        device = torch.device('cuda:0'),
        name = 'unnamed',
        seed: int = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self._items_dist = torch.FloatTensor(items_count) / sum(items_count)

    def select_action(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        return self._items_dist.multinomial(
            batch_size,
            replacement=True,
            generator=self._torch_generator
        ).to(self._device)
    
    def action_dist(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        return self._items_dist.repeat(batch_size, 1).to(self._device)

# @dataclass
# class BaseLinPolicy(BaseContextualPolicy):
#     """Base class for contextual bandit policies using linear regression.

#     Parameters
#     ------------
#     dim: int
#         Number of dimensions of context vectors.

#     n_actions: int
#         Number of actions.

#     len_list: int, default=1
#         Length of a list of actions in a recommendation/ranking inferface, slate size.
#         When Open Bandit Dataset is used, 3 should be set.

#     batch_size: int, default=1
#         Number of samples used in a batch parameter update.

#     random_state: int, default=None
#         Controls the random seed in sampling actions.

#     epsilon: float, default=0.
#         Exploration hyperparameter that must take value in the range of [0., 1.].

#     """

#     def __post_init__(self) -> None:
#         """Initialize class."""
#         super().__post_init__()
#         self.theta_hat = np.zeros((self.dim, self.n_actions))
#         self.A_inv = np.concatenate(
#             [np.identity(self.dim) for _ in np.arange(self.n_actions)]
#         ).reshape(self.n_actions, self.dim, self.dim)
#         self.b = np.zeros((self.dim, self.n_actions))

#         self.A_inv_temp = np.concatenate(
#             [np.identity(self.dim) for _ in np.arange(self.n_actions)]
#         ).reshape(self.n_actions, self.dim, self.dim)
#         self.b_temp = np.zeros((self.dim, self.n_actions))

#     def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
#         """Update policy parameters.

#         Parameters
#         ------------
#         action: int
#             Selected action by the policy.

#         reward: float
#             Observed reward for the chosen action and position.

#         context: array-like, shape (1, dim_context)
#             Observed context vector.

#         """
#         self.n_trial += 1
#         self.action_counts[action] += 1
#         # update the inverse matrix by the Woodbury formula
#         self.A_inv_temp[action] -= (
#             self.A_inv_temp[action]
#             @ context.T
#             @ context
#             @ self.A_inv_temp[action]
#             / (1 + context @ self.A_inv_temp[action] @ context.T)[0][0]
#         )
#         self.b_temp[:, action] += reward * context.flatten()
#         if self.n_trial % self.batch_size == 0:
#             self.A_inv, self.b = (
#                 np.copy(self.A_inv_temp),
#                 np.copy(self.b_temp),
#             )

#     @abc.abstractmethod
#     def select_action(self, context: np.ndarray) -> np.ndarray:
#         raise NotImplementedError()

#     def compute_batch_action_dist(self, context : np.ndarray):
#         n_rounds = context.shape[0]
#         actions = np.zeros((n_rounds, self.n_actions, self.len_list))

#         for i in range(n_rounds):
#             selected_actions = self.select_action(context=context[i].reshape(1, -1))
#             actions[i, selected_actions, np.arange(self.len_list)] = 1

#         return actions



# @dataclass
# class LinEpsilonGreedy(BaseLinPolicy):
#     """Linear Epsilon Greedy.

#     Parameters
#     ------------
#     dim: int
#         Number of dimensions of context vectors.

#     n_actions: int
#         Number of actions.

#     len_list: int, default=1
#         Length of a list of actions in a recommendation/ranking inferface, slate size.
#         When Open Bandit Dataset is used, 3 should be set.

#     batch_size: int, default=1
#         Number of samples used in a batch parameter update.

#     n_trial: int, default=0
#         Current number of trials in a bandit simulation.

#     random_state: int, default=None
#         Controls the random seed in sampling actions.

#     epsilon: float, default=0.
#         Exploration hyperparameter that must take value in the range of [0., 1.].

#     References
#     ------------
#     L. Li, W. Chu, J. Langford, and E. Schapire.
#     A contextual-bandit approach to personalized news article recommendation.
#     In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

#     """

#     epsilon: float = 0.0

#     def __post_init__(self) -> None:
#         """Initialize class."""
#         check_scalar(self.epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
#         self.policy_name = f"linear_epsilon_greedy_{self.epsilon}"

#         super().__post_init__()

#     def select_action(self, context: np.ndarray) -> np.ndarray:
#         """Select action for new data.

#         Parameters
#         ------------
#         context: array-like, shape (1, dim_context)
#             Observed context vector.

#         Returns
#         ----------
#         selected_actions: array-like, shape (len_list, )
#             List of selected actions.

#         """
#         check_array(array=context, name="context", expected_dim=2)
#         if context.shape[0] != 1:
#             raise ValueError("Expected `context.shape[0] == 1`, but found it False")

#         if self.random_.rand() > self.epsilon:
#             self.theta_hat = np.concatenate(
#                 [
#                     self.A_inv[i] @ self.b[:, i][:, np.newaxis]
#                     for i in np.arange(self.n_actions)
#                 ],
#                 axis=1,
#             )  # dim * n_actions
#             predicted_rewards = (context @ self.theta_hat).flatten()
#             return predicted_rewards.argsort()[::-1][: self.len_list]
#         else:
#             return self.random_.choice(
#                 self.n_actions, size=self.len_list, replace=False
#             )


# @dataclass
# class LinUCB(BaseLinPolicy):
#     """Linear Upper Confidence Bound.

#     Parameters
#     ----------
#     dim: int
#         Number of dimensions of context vectors.

#     n_actions: int
#         Number of actions.

#     len_list: int, default=1
#         Length of a list of actions in a recommendation/ranking inferface, slate size.
#         When Open Bandit Dataset is used, 3 should be set.

#     batch_size: int, default=1
#         Number of samples used in a batch parameter update.

#     random_state: int, default=None
#         Controls the random seed in sampling actions.

#     epsilon: float, default=0.
#         Exploration hyperparameter that must be greater than or equal to 0.0.

#     References
#     --------------
#     L. Li, W. Chu, J. Langford, and E. Schapire.
#     A contextual-bandit approach to personalized news article recommendation.
#     In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

#     """

#     epsilon: float = 0.0

#     def __post_init__(self) -> None:
#         """Initialize class."""
#         check_scalar(self.epsilon, "epsilon", float)
#         self.policy_name = f"linear_ucb_{self.epsilon}"

#         super().__post_init__()

#     def select_action(self, context: np.ndarray) -> np.ndarray:
#         """Select action for new data.

#         Parameters
#         ----------
#         context: array
#             Observed context vector.

#         Returns
#         ----------
#         selected_actions: array-like, shape (len_list, )
#             List of selected actions.

#         """
#         check_array(array=context, name="context", expected_dim=2)
#         if context.shape[0] != 1:
#             raise ValueError("Expected `context.shape[0] == 1`, but found it False")

#         self.theta_hat = np.concatenate(
#             [
#                 self.A_inv[i] @ self.b[:, i][:, np.newaxis]
#                 for i in np.arange(self.n_actions)
#             ],
#             axis=1,
#         )  # dim * n_actions
#         sigma_hat = np.concatenate(
#             [
#                 np.sqrt(context @ self.A_inv[i] @ context.T)
#                 for i in np.arange(self.n_actions)
#             ],
#             axis=1,
#         )  # 1 * n_actions
#         ucb_scores = (context @ self.theta_hat + self.epsilon * sigma_hat).flatten()
#         return ucb_scores.argsort()[::-1][: self.len_list]

#     def predict_scores(self, context: np.ndarray) -> np.ndarray:

#         check_array(array=context, name="context", expected_dim=2)
#         if context.shape[0] != 1:
#             raise ValueError("Expected `context.shape[0] == 1`, but found it False")

#         self.theta_hat = np.concatenate(
#             [
#                 self.A_inv[i] @ self.b[:, i][:, np.newaxis]
#                 for i in np.arange(self.n_actions)
#             ],
#             axis=1,
#         )  # dim * n_actions
#         sigma_hat = np.concatenate(
#             [
#                 np.sqrt(context @ self.A_inv[i] @ context.T)
#                 for i in np.arange(self.n_actions)
#             ],
#             axis=1,
#         )  # 1 * n_actions
#         ucb_scores = (context @ self.theta_hat + self.epsilon * sigma_hat).flatten()
        
#         return ucb_scores


# @dataclass
# class LinTS(BaseLinPolicy):
#     """Linear Thompson Sampling.

#     Parameters
#     ----------
#     dim: int
#         Number of dimensions of context vectors.

#     n_actions: int
#         Number of actions.

#     len_list: int, default=1
#         Length of a list of actions in a recommendation/ranking inferface, slate size.
#         When Open Bandit Dataset is used, 3 should be set.

#     batch_size: int, default=1
#         Number of samples used in a batch parameter update.

#     random_state: int, default=None
#         Controls the random seed in sampling actions.

#     """

#     def __post_init__(self) -> None:
#         """Initialize class."""
#         self.policy_name = "linear_ts"

#         super().__post_init__()

#     def select_action(self, context: np.ndarray) -> np.ndarray:
#         """Select action for new data.

#         Parameters
#         ----------
#         context: array-like, shape (1, dim_context)
#             Observed context vector.

#         Returns
#         ----------
#         selected_actions: array-like, shape (len_list, )
#             List of selected actions.

#         """
#         self.theta_hat = np.concatenate(
#             [
#                 self.A_inv[i] @ self.b[:, i][:, np.newaxis]
#                 for i in np.arange(self.n_actions)
#             ],
#             axis=1,
#         )
#         theta_sampled = np.concatenate(
#             [
#                 self.random_.multivariate_normal(self.theta_hat[:, i], self.A_inv[i])[
#                     :, np.newaxis
#                 ]
#                 for i in np.arange(self.n_actions)
#             ],
#             axis=1,
#         )

#         predicted_rewards = (context @ theta_sampled).flatten()
#         return predicted_rewards.argsort()[::-1][: self.len_list]