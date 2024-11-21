from abc import abstractmethod
from tqdm import tqdm

from typing import List
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


class AbstractDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def state_dim(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_users(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def num_items(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def items_count(self):
        raise NotImplementedError()

    @abstractmethod
    def iterate_dataset(self):
        raise NotImplementedError()


class MovieLens(AbstractDataset):
    def __init__(
        self,
        num_samples: int,
        file_path: str = './data/ml-1m.zip',
    ):
        super().__init__()


class MovieLensBasicMDP(AbstractDataset):
    def __init__(
        self,
        num_items: int,
        num_samples: int,
        user_sequences: List[List[int]]
    ):
        super().__init__()

        self._num_items = num_items
        self._num_samples = num_samples
        self._num_users = len(user_sequences)
        self._items_count = np.unique(np.concatenate(user_sequences), return_counts=True)[1]

        if num_samples > self._num_users:
            raise ValueError('num samples must be <= number of users')

        self._user_sequences = []
        self._seq_lengths = []
        for seq in user_sequences:
            if len(seq) <= 1:
                raise ValueError('sequence length must be > 1 for every user')

            self._user_sequences.append(np.array(seq))
            self._seq_lengths.append(len(seq))
        
        # prefetch every (s,a) pair for evaluation
        states_acitons = []
        for seq in user_sequences:
            for i, (a_prev, a) in enumerate(zip(seq[:-1], seq[1:])):
                states_acitons.append([a_prev, a])

        self._states_acitons = np.array(states_acitons)

    @property
    def state_dim(self):
        return self._num_items

    @property
    def num_users(self):
        return self._num_users
    
    @property
    def num_items(self):
        return self._num_items

    @property
    def items_count(self):
        return self._items_count

    def generate(self):
        numpy_generator = np.random.default_rng(get_worker_info().seed)

        while True:
            first_states = np.zeros(self._num_samples)
            current_states = np.zeros(self._num_samples)
            actions = np.zeros(self._num_samples)
            next_states = np.zeros(self._num_samples)
            step_num = np.zeros(self._num_samples)
            has_next = np.ones(self._num_samples)

            user_idx = numpy_generator.choice(
                self._num_users, self._num_samples, replace=False)

            for i, idx in enumerate(user_idx):
                seq = self._user_sequences[idx]
                pos = numpy_generator.integers(0, self._seq_lengths[idx]-1)

                first_states[i] = seq[0]
                current_states[i] = seq[pos]
                actions[i] = seq[pos+1]
                next_states[i] = seq[pos+1]
                step_num[i] = pos
                if (pos+2) == self._seq_lengths[idx]:
                    has_next[i] = 0.0

            first_states = torch.LongTensor(first_states)
            current_states = torch.LongTensor(current_states)
            next_states = torch.LongTensor(next_states)

            yield (
                F.one_hot(first_states, self._num_items).float(),
                F.one_hot(current_states, self._num_items).float(),
                torch.LongTensor(actions),
                F.one_hot(next_states, self._num_items).float(),
                torch.ones(self._num_samples).float(),
                torch.LongTensor(step_num),
                torch.FloatTensor(has_next)
            )

    def __iter__(self):
        return iter(self.generate())

    def iterate_dataset(self, batch_size: int):
        data_size = self._states_acitons.shape[0]

        s = torch.LongTensor(self._states_acitons[:, 0])
        a = torch.LongTensor(self._states_acitons[:, 1])

        for i in range(0, data_size, batch_size):
            idx = torch.arange(start=i, end=min(i+batch_size, data_size))
            states = F.one_hot(s[idx], self._num_items).float()
            actions = a[idx].long()
            rewards = torch.ones(actions.shape[0]).float()

            yield states, actions, rewards


class MovieLensSasrecMDP(AbstractDataset):
    def __init__(
        self,
        num_items: int,
        num_samples: int,
        user_sequences: List[List[int]],
        sasrec_path: str,
        sasrec_device: torch.device = torch.device('cpu'),
        embeddings_path: str = None
    ):
        super().__init__()

        self._num_items = num_items
        self._num_samples = num_samples
        self._num_users = len(user_sequences)
        self._items_count = np.unique(np.concatenate(user_sequences), return_counts=True)[1]

        if num_samples > self._num_users:
            raise ValueError('num samples must be <= number of users')

        # load sasrec
        if embeddings_path is None:
            sasrec = torch.load(sasrec_path).to(sasrec_device)
            sasrec.eval()

        self._user_sequences = []
        self._seq_lengths = []
        self._states = []
        for seq in tqdm(user_sequences):
            if len(seq) <= 1:
                raise ValueError('sequence length must be > 1 for every user')

            self._user_sequences.append(np.array(seq))
            self._seq_lengths.append(len(seq))

            if embeddings_path is not None:
                continue

            with torch.no_grad():
                user_states = []
                for i in range(1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(sasrec_device)
                    user_states.append(sasrec.score_with_state(s)[-1].cpu())
            
            self._states.append(user_states)

        if embeddings_path is not None:
            self._states = torch.load(embeddings_path)
        
        # prefetch every (s,a) pair for evaluation
        eval_acitons = []
        eval_states = []
        for u, seq in enumerate(user_sequences):
            for i in range(1, len(seq)):
                eval_acitons.append(seq[i])
                eval_states.append(self._states[u][i-1])

        self._eval_actions = np.array(eval_acitons)
        self._eval_states = torch.stack(eval_states)

    @property
    def state_dim(self):
        return self._states[0][0].shape[0]

    @property
    def num_users(self):
        return self._num_users
    
    @property
    def num_items(self):
        return self._num_items

    @property
    def items_count(self):
        return self._items_count

    def generate(self):
        numpy_generator = np.random.default_rng(get_worker_info().seed)

        while True:
            first_states = []
            current_states = []
            actions = np.zeros(self._num_samples)
            next_states = []
            step_num = np.zeros(self._num_samples)
            has_next = np.ones(self._num_samples)

            user_idx = numpy_generator.choice(
                self._num_users, self._num_samples, replace=False)
            
            for i, idx in enumerate(user_idx):
                seq = self._user_sequences[idx]
                pos = numpy_generator.integers(0, self._seq_lengths[idx]-1)

                first_states.append(self._states[idx][0])
                current_states.append(self._states[idx][pos])
                next_states.append(self._states[idx][pos+1])
                actions[i] = seq[pos+1]
                step_num[i] = pos
                if (pos+2) == self._seq_lengths[idx]:
                    has_next[i] = 0.0

            yield (
                torch.stack(first_states),
                torch.stack(current_states),
                torch.LongTensor(actions),
                torch.stack(next_states),
                torch.ones(self._num_samples).float(),
                torch.LongTensor(step_num),
                torch.FloatTensor(has_next)
            )

    def __iter__(self):
        return iter(self.generate())

    def iterate_dataset(self, batch_size: int):
        data_size = self._eval_actions.shape[0]

        s = self._eval_states
        a = torch.LongTensor(self._eval_actions)

        for i in range(0, data_size, batch_size):
            idx = torch.arange(start=i, end=min(i+batch_size, data_size))
            states = s[idx]
            actions = a[idx].long()
            rewards = torch.ones(actions.shape[0]).float()

            yield states, actions, rewards
