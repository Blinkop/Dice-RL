from abc import abstractmethod

from typing import List
import numpy as np
import pandas as pd

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
    def action_dim(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def action_embs(self):
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
        states_path: str,
        predictions_path: str,
        action_embeddings_path: str,
        data_path: str = './data/ml-1m.zip',
    ):
        super().__init__()

        self._num_samples = num_samples

        trainset, testset, holdout, description = MovieLens.create_dataset(path=data_path)

        self._num_items = description['n_items']
        self._train_df = trainset
        self._test_df = testset
        self._test_holdout_df = holdout

        self._items_count = np.zeros(self._num_items)
        items_unique, items_count = np.unique(self._train_df['itemid'], return_counts=True)
        for item, count in zip(items_unique, items_count):
            self._items_count[item] = count

        self._user_sequences = MovieLens.create_sequences(
            testset=self._test_df, holdout=self._test_holdout_df)
        self._user_ids = list(self._user_sequences.keys())

        if self._num_samples > len(self._user_sequences):
            raise ValueError('num samples must be <= number of valid users')

        self._states = torch.load(states_path)
        self._predictions = torch.load(predictions_path)
        self._action_embs = torch.load(action_embeddings_path)
        self._precalc_validation()

        eval_states = []
        eval_acitons = []
        for u, seq in self._user_sequences.items():
            for i in range(1, len(seq)):
                eval_states.append(self._states[u][i-1])
                eval_acitons.append(seq[i])

        self._eval_states = torch.stack(eval_states)
        self._eval_actions = torch.LongTensor(eval_acitons)

    def _precalc_validation(self):
        if len(self._states) != len(self._user_sequences):
            raise ValueError(f'number of users mismatch in preloaded states')
        if len(self._predictions) != len(self._user_sequences):
            raise ValueError(f'number of users mismatch in preloaded predictions')
        if self._action_embs.shape[0] != self._num_items:
            raise ValueError(f'number of items in preloaded action embeddings mismatch')

        for u in self._user_sequences:
            if len(self._user_sequences[u]) != len(self._states[u]):
                raise ValueError(f'sequence length of user {u} mismatch in preloaded states')
            if len(self._user_sequences[u]) != len(self._predictions[u]):
                raise ValueError(f'sequence length of user {u} mismatch in preloaded predictions')

    @staticmethod
    def create_dataset(path: str):
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
            data_path=path,
            splitting='temporal_full',
            q=0.8
        )

        return trainset, testset, holdout, data_description

    @staticmethod
    def create_sequences(testset: pd.DataFrame, holdout: pd.DataFrame):
        user_sequences = {}

        sequences = testset.sort_values(['userid', 'timestamp'])\
            .groupby('userid', sort=False)['itemid']\
            .apply(list)

        for i, seq in sequences.items():
            user_sequences[i] = seq

        sequences = holdout.sort_values(['userid', 'timestamp'])\
            .groupby('userid', sort=False)['itemid']\
            .apply(list)
        
        for i, seq in sequences.items():
            if i in user_sequences:
                user_sequences[i] += seq

        for _, seq in user_sequences.items():
            if len(seq) == 1:
                raise ValueError('There must be at least two items in a presented sequences')
            
        return user_sequences

    @property
    def state_dim(self):
        return self._states[self._user_ids[0]][0].shape[0]
    
    @property
    def action_dim(self):
        return self._action_embs.shape[1]
    
    @property
    def action_embs(self):
        return self._action_embs

    @property
    def num_users(self):
        return len(self._user_sequences)
    
    @property
    def num_items(self):
        return self._num_items
    
    @property
    def items_count(self):
        return self._items_count

    def _generate(self):
        numpy_generator = np.random.default_rng(get_worker_info().seed)

        while True:
            first_states = []
            first_sampled_actions = []
            current_states = []
            actions = np.zeros(self._num_samples)
            next_states = []
            next_sampled_actions = []
            step_num = np.zeros(self._num_samples)
            has_next = np.ones(self._num_samples)

            user_idx = numpy_generator.choice(self._user_ids, self._num_samples, replace=False)
            
            for i, idx in enumerate(user_idx):
                seq = self._user_sequences[idx]
                pos = numpy_generator.integers(0, len(seq)-1)

                first_states.append(self._states[idx][0])
                first_sampled_actions.append(self._predictions[idx][0])
                current_states.append(self._states[idx][pos])
                actions[i] = seq[pos+1]
                next_states.append(self._states[idx][pos+1])
                next_sampled_actions.append(self._predictions[idx][pos+1])
                step_num[i] = pos
                if (pos+2) == len(seq):
                    has_next[i] = 0.0

            yield (
                torch.stack(first_states),
                torch.LongTensor(first_sampled_actions).reshape(-1, 1),
                torch.stack(current_states),
                torch.LongTensor(actions),
                torch.stack(next_states),
                torch.LongTensor(next_sampled_actions).reshape(-1, 1),
                torch.ones(self._num_samples).float(),
                torch.LongTensor(step_num),
                torch.FloatTensor(has_next)
            )
    
    def __iter__(self):
        return iter(self._generate())
    
    def iterate_dataset(self, batch_size: int):
        data_size = self._eval_actions.shape[0]

        for i in range(0, data_size, batch_size):
            idx = torch.arange(start=i, end=min(i+batch_size, data_size))
            states = self._eval_states[idx]
            actions = self._eval_actions[idx]
            rewards = torch.ones(actions.shape[0]).float()

            yield states, actions, rewards
