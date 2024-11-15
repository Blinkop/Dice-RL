from typing import List
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


class MovieLensBasicMDP(IterableDataset):
    def __init__(
        self,
        num_items: int,
        user_sequences: List[List[int]],
        num_samples: int
    ):
        super().__init__()

        self._num_items = num_items
        self._num_samples = num_samples

        first_items = torch.tensor([s[0] for s in user_sequences])
        
        states_acitons = []
        step_num = []
        # for a case of different MDP with full sequence
        # for seq in user_sequences:
        #     for i in range(1, len(seq)):
        #         states_acitons.append((seq[:i], seq[i]))
        for seq in user_sequences:
            for i, (a_prev, a) in enumerate(zip(seq[:-1], seq[1:])):
                states_acitons.append([a_prev, a])
                step_num.append(i)

        self.first_items = np.array(first_items)
        self.states_acitons = np.array(states_acitons)
        self.step_num = np.array(step_num)

    def generate(self):
        while True:
            first_item_idx = np.random.choice(
                self.first_items.shape[0], self._num_samples, replace=True)
            state_action_idx = np.random.choice(
                self.states_acitons.shape[0], self._num_samples, replace=False)

            first_items = torch.LongTensor(self.first_items[first_item_idx])
            states_actions = torch.LongTensor(self.states_acitons[state_action_idx])
            step_num = torch.LongTensor(self.step_num[state_action_idx])

            first_states = F.one_hot(first_items, self._num_items).float()
            states = F.one_hot(states_actions[:, 0], self._num_items).float()
            actions = states_actions[:, 1].long()
            next_states = F.one_hot(actions, self._num_items).float()
            rewards = torch.ones(states.shape[0]).float()

            yield first_states, states, actions, next_states, rewards, step_num

    def __iter__(self):
        return iter(self.generate())

    def iterate_dataset(self, batch_size: int):
        data_size = self.states_acitons.shape[0]

        s = torch.LongTensor(self.states_acitons[:, 0])
        a = torch.LongTensor(self.states_acitons[:, 1])

        for i in range(0, data_size, batch_size):
            idx = torch.arange(start=i, end=min(i+batch_size, data_size))
            states = F.one_hot(s[idx], self._num_items).float()
            actions = a[idx].long()
            rewards = torch.ones(states.shape[0]).float()

            yield states, actions, rewards
