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


class RandomPolicy(Policy):
    def __init__(
        self,
        num_actions: int,
        device = torch.device('cuda:0'),
        name = 'unnamed'
    ):
        super().__init__(device=device, name=name)

        self._num_actions = num_actions

    def select_action(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        return torch.randint(
            0, self._num_actions, size=(batch_size,)
        ).to(self._device)
    
    def action_dist(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        action_dist = (1 / self._num_actions) * torch.ones((batch_size, self._num_actions))

        return action_dist.to(self._device)


class PopularRandomPolicy(Policy):
    def __init__(
        self,
        items_count: List[int],
        device = torch.device('cuda:0'),
        name = 'unnamed'
    ):
        super().__init__(device=device, name=name)

        self._items_dist = torch.FloatTensor(items_count) / sum(items_count)

    def select_action(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        return self._items_dist\
            .multinomial(batch_size, replacement=True)\
            .to(self._device)
    
    def action_dist(self, state: torch.Tensor):
        batch_size = self._get_batch_size(state)

        return self._items_dist.repeat(batch_size, 1).to(self._device)
