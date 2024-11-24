import abc
from typing import List

import torch
import torch.nn.functional as F

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


class SASRec(Policy):
    def __init__(
        self,
        model_path: str,
        num_items: int,
        device=torch.device('cuda:0'),
        name='sasrec',
    ):
        super().__init__(device=device, name=name)

        self._num_items = num_items
        self._sasrec = torch.load(model_path).to(device)

    @torch.no_grad()
    def select_action(self, state: List):
        actions = []
        for s in state:
            logits = self._sasrec.score_with_state(s)[0].flatten().detach()
            actions.append(logits.argmax().item())

        return torch.LongTensor(actions).to(self._device)
    
    def action_dist(self, state: List):
        actions = []
        for s in state:
            logits = self._sasrec.score_with_state(s)[0].flatten().detach()
            actions.append(logits.argmax().item())

        actions = torch.LongTensor(actions)

        return F.one_hot(actions, self._num_items).float().to(self._device)


class DT4RecPolicy(Policy):
    def __init__(
        self,
        model_path: str,
        device=torch.device("cuda:0"),
        name="unnamed"
    ):
        super().__init__(device=device, name=name)

        self._model = torch.load(model_path).head.to(device)
        self._model.eval()

    def select_action(self, state):
        probs = torch.softmax(self.model(state), dim=1)
        actions = torch.multinomial(probs, 1)

        return actions

    def action_dist(self, state):
        probs = torch.softmax(self.model(state), dim=1)

        return probs
    

class CQLSASRec(Policy):
    def __init__(
        self,
        model_path: str,
        device=torch.device("cuda:0"),
        name="unnamed"
    ):
        super().__init__(device=device, name=name)

        self._trainer = torch.load(model_path)
        self._trainer.q_1 = self._trainer.q_1.to(device)
        self._trainer.q_2 = self._trainer.q_2.to(device)
        self._trainer.body = self._trainer.body.to(device)
        self._trainer.q_1.eval()
        self._trainer.q_2.eval()
        self._trainer.body.eval()

    def select_action(self, state):
        actions = []
        for s in state:
            body_out = self._trainer.body.score_with_state(s)[-1]
            body_out = body_out.reshape(-1, body_out.shape[-1])
            out = (self._trainer.q_1(body_out) + self._trainer.q_2(body_out)) / 2.0
            actions.append(out.argmax().item())

        return torch.LongTensor(actions).to(self._device)

    def action_dist(self, state):
        raise NotImplementedError()


class Precalc(Policy):
    def __init__(
        self,
        device=torch.device('cuda:0'),
        name='precalc',
    ):
        super().__init__(device=device, name=name)

    def select_action(self, state):
        return state.to(self._device)
    
    def action_dist(self, state):
        return state.to(self._device)
