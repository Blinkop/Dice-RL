from abc import abstractmethod

from typing import List
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from .data_utils import get_dataset


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


class MovieLensBase(AbstractDataset):
    def __init__(
        self,
        num_samples: int,
        policy: str,
        file_path: str = './data/ml-1m.zip',
        predictions_path: str = None
    ):
        super().__init__()

        self._num_samples = num_samples
        self._policy = policy

        if self._policy not in ['random', 'poprandom', 'sasrec', 'precalc']:
            raise ValueError(f'unknown policy "{self._policy}"')
        
        (
            trainset,
            data_description,
            _,
            testset,
            _,
            holdout
        ) = get_dataset(
            validation_size=1024,
            test_size=5000,
            data_path=file_path,
            splitting='temporal_full',
            q=0.8
        )

        self._num_items = data_description['n_items']
        self._train_df = trainset
        self._test_df = testset
        self._test_holdout_df = holdout

        self._items_count = np.zeros(self.num_items)
        items_unique, items_count = np.unique(self._train_df['itemid'], return_counts=True)
        for item, count in zip(items_unique, items_count):
            self._items_count[item] = count

        self._user_sequences = self._create_sequences()
        self._user_ids = list(self._user_sequences.keys())

        if self._num_samples > len(self._user_sequences):
            raise ValueError('num samples must be <= number of valid users')

        if policy == 'precalc':
            self._predictions_file = torch.load(predictions_path)

        eval_acitons = []
        for _, seq in self._user_sequences.items():
            for i in range(1, len(seq)):
                eval_acitons.append(seq[i])

        self._eval_actions = np.array(eval_acitons)

    @property
    def num_users(self):
        return len(self._user_sequences)
    
    @property
    def num_items(self):
        return self._num_items
    
    @property
    def items_count(self):
        return self._items_count
    
    def _create_sequences(self):
        user_sequences = {}

        sequences = self._test_df.sort_values(['userid', 'timestamp'])\
            .groupby('userid', sort=False)['itemid']\
            .apply(list)

        for i, seq in sequences.items():
            user_sequences[i] = seq

        sequences = self._test_holdout_df.sort_values(['userid', 'timestamp'])\
            .groupby('userid', sort=False)['itemid']\
            .apply(list)
        
        for i, seq in sequences.items():
            if i in user_sequences:
                user_sequences[i] += seq

        for _, seq in user_sequences.items():
            if len(seq) == 1:
                raise ValueError('There must be at least two items in a presented sequences')
            
        return user_sequences
    
    def _create_policy_input(self, uid: int, pos: int, seq: List[int]):
        if self._policy in ['random', 'poprandom']:
            return None
        elif self._policy == 'sasrec':
            return torch.LongTensor(seq[:pos+1])
        elif self._policy == 'precalc':
            pred = self._predictions_file[uid][pos]
            return pred if torch.is_tensor(pred) else torch.tensor(pred).long()

    @abstractmethod
    def _generate(self):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._generate())


class MovieLensBasicMDP(MovieLensBase):
    def __init__(
        self,
        num_samples: int,
        policy: str,
        file_path: str = './data/ml-1m.zip',
        predictions_path: str = None
    ):
        super().__init__(
            num_samples=num_samples,
            policy=policy,
            file_path=file_path,
            predictions_path=predictions_path
        )

        eval_states = []
        for _, seq in self._user_sequences.items():
            for i in range(1, len(seq)):
                eval_states.append(seq[i-1])

        self._eval_states = np.array(eval_states)

    @property
    def state_dim(self):
        return self.num_items
    
    def _generate(self):
        numpy_generator = np.random.default_rng(get_worker_info().seed)

        while True:
            first_states = np.zeros(self._num_samples)
            first_inputs = []
            current_states = np.zeros(self._num_samples)
            actions = np.zeros(self._num_samples)
            next_states = np.zeros(self._num_samples)
            next_inputs = []
            step_num = np.zeros(self._num_samples)
            has_next = np.ones(self._num_samples)

            user_idx = numpy_generator.choice(self._user_ids, self._num_samples, replace=False)
            
            for i, idx in enumerate(user_idx):
                seq = self._user_sequences[idx]
                pos = numpy_generator.integers(0, len(seq)-1)

                first_states[i] = seq[0]
                first_inputs.append(self._create_policy_input(idx, 0, seq))
                current_states[i] = seq[pos]
                actions[i] = seq[pos+1]
                next_states[i] = seq[pos+1]
                next_inputs.append(self._create_policy_input(idx, pos+1, seq))
                step_num[i] = pos
                if (pos+2) == len(seq):
                    has_next[i] = 0.0

            first_states = torch.LongTensor(first_states)
            current_states = torch.LongTensor(current_states)
            next_states = torch.LongTensor(next_states)

            if self._policy == 'precalc':
                first_inputs = torch.stack(first_inputs)
                next_inputs = torch.stack(next_inputs)

            yield (
                F.one_hot(first_states, self.num_items).float(),
                first_inputs,
                F.one_hot(current_states, self.num_items).float(),
                torch.LongTensor(actions),
                F.one_hot(next_states, self.num_items).float(),
                next_inputs,
                torch.ones(self._num_samples).float(),
                torch.LongTensor(step_num),
                torch.FloatTensor(has_next)
            )

    def iterate_dataset(self, batch_size: int):
        data_size = self._eval_actions.shape[0]
        
        s = torch.LongTensor(self._eval_states)
        a = torch.LongTensor(self._eval_actions)

        for i in range(0, data_size, batch_size):
            idx = torch.arange(start=i, end=min(i+batch_size, data_size))
            states = F.one_hot(s[idx], self._num_items).float()
            actions = a[idx].long()
            rewards = torch.ones(actions.shape[0]).float()

            yield states, actions, rewards


class MovieLensSasrecMDP(MovieLensBase):
    def __init__(
        self,
        num_samples: int,
        policy: str,
        file_path: str = './data/ml-1m.zip',
        predictions_path: str = None,
        states_path: str = './models/sasrec_ml_states.pt'
    ):
        super().__init__(
            num_samples=num_samples,
            policy=policy,
            file_path=file_path,
            predictions_path=predictions_path
        )

        self._states = torch.load(states_path)

        eval_states = []
        for u, seq in self._user_sequences.items():
            for i in range(1, len(seq)):
                eval_states.append(self._states[u][i-1])

        self._eval_states = torch.stack(eval_states)

    @property
    def state_dim(self):
        return self._states[self._user_ids[0]][0].shape[0]
    
    def _generate(self):
        numpy_generator = np.random.default_rng(get_worker_info().seed)

        while True:
            first_states = []
            first_inputs = []
            current_states = []
            actions = np.zeros(self._num_samples)
            next_states = []
            next_inputs = []
            step_num = np.zeros(self._num_samples)
            has_next = np.ones(self._num_samples)

            user_idx = numpy_generator.choice(self._user_ids, self._num_samples, replace=False)
            
            for i, idx in enumerate(user_idx):
                seq = self._user_sequences[idx]
                pos = numpy_generator.integers(0, len(seq)-1)

                first_states.append(self._states[idx][0])
                first_inputs.append(self._create_policy_input(idx, 0, seq))
                current_states.append(self._states[idx][pos])
                next_states.append(self._states[idx][pos+1])
                next_inputs.append(self._create_policy_input(idx, pos+1, seq))
                actions[i] = seq[pos+1]
                step_num[i] = pos
                if (pos+2) == len(seq):
                    has_next[i] = 0.0

            if self._policy == 'precalc':
                first_inputs = torch.stack(first_inputs)
                next_inputs = torch.stack(next_inputs)

            yield (
                torch.stack(first_states),
                first_inputs,
                torch.stack(current_states),
                torch.LongTensor(actions),
                torch.stack(next_states),
                next_inputs,
                torch.ones(self._num_samples).float(),
                torch.LongTensor(step_num),
                torch.FloatTensor(has_next)
            )

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