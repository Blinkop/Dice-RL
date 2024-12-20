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

    def _get_batch_size(self, state):
        if isinstance(state, list):
            return len(state)
        elif torch.is_tensor(state):
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
        name = 'random',
        seed: int = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self._num_actions = num_actions

    def select_action(self, state):
        batch_size = self._get_batch_size(state)

        return torch.randint(
            0,
            self._num_actions,
            size=(batch_size,),
            generator=self._torch_generator
        ).to(self._device)
    
    def action_dist(self, state):
        batch_size = self._get_batch_size(state)

        action_dist = (1 / self._num_actions) * torch.ones((batch_size, self._num_actions))

        return action_dist.to(self._device)


class PopularRandomPolicy(AbstractRandomPolicy):
    def __init__(
        self,
        items_count: List[int],
        device = torch.device('cuda:0'),
        name = 'poprandom',
        seed: int = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self._items_dist = torch.FloatTensor(items_count) / sum(items_count)

    def select_action(self, state):
        batch_size = self._get_batch_size(state)

        return self._items_dist.multinomial(
            batch_size,
            replacement=True,
            generator=self._torch_generator
        ).to(self._device)
    
    def action_dist(self, state):
        batch_size = self._get_batch_size(state)

        return self._items_dist.repeat(batch_size, 1).to(self._device)
