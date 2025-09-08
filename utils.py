import inspect
from typing import Optional, List

import numpy as np

import torch
from torch.utils.data import IterableDataset, get_worker_info, default_collate


class DiceDatasetWrapper(IterableDataset):
    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[np.ndarray],
        target_actions: List[np.ndarray],
        batch_size: int
    ) -> None:
        super().__init__()

        trajectory_len = [len(r) for r in rewards]
        trajectory_offset = np.cumsum([0] + trajectory_len)[:-1]

        self._num_trajectories = len(rewards)
        self._trajectory_len = torch.tensor(trajectory_len, dtype=torch.long)
        self._trajectory_offset = torch.tensor(trajectory_offset, dtype=torch.long)

        self._batch_size = batch_size

        self._state = torch.tensor(np.concatenate(states), dtype=torch.float)
        self._action = torch.tensor(np.concatenate(actions), dtype=torch.long)
        self._reward = torch.tensor(np.concatenate(rewards), dtype=torch.float)
        self._target_action = torch.tensor(np.concatenate(target_actions), dtype=torch.long)

    def generate(self):
        generator = torch.Generator()
        generator.manual_seed(get_worker_info().seed)

        while True:
            idx = torch.randint(
                self._num_trajectories,
                size=(self._batch_size,),
                generator=generator
            )
            t = torch.tensor([
                torch.randint(0, l - 2, size=(1,), generator=generator)
                for l in self._trajectory_len[idx]
            ], dtype=torch.long)

            flatten_idx = self._trajectory_offset[idx]
            flatten_t = flatten_idx + t

            yield (
                self._state[flatten_idx],
                self._target_action[flatten_idx],
                self._state[flatten_t],
                self._action[flatten_t],
                self._reward[flatten_t],
                self._state[flatten_t + 1],
                self._target_action[flatten_t + 1]
            )
    
    def __iter__(self):
        return iter(self.generate())

    
def custom_collate(batch_list):
    default_batch = default_collate(batch_list)

    return [tensor.flatten(0, 1) for tensor in default_batch]


def get_lambda_code(f):
    return inspect.getsourcelines(f)[0][0]\
        .strip("['\\n']").split(" = ")[1]


def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
    expected_dtype: Optional[type] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> ValueError:
    """Input validation on array.

    Parameters
    -------
    array: object
        Input array to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Expected dimension of the input array.

    expected_dtype: {type, tuple of type}, default=None
        Expected dtype of the input array.

    min_val: float, default=None
        Minimum value allowed in the input array.

    max_val: float, default=None
        Maximum value allowed in the input array.

    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be {expected_dim}D array, but got {type(array)}")
    if array.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D array, but got {array.ndim}D array"
        )
    if expected_dtype is not None:
        if not np.issubsctype(array, expected_dtype):
            raise ValueError(
                f"The elements of {name} must be {expected_dtype}, but got {array.dtype}"
            )
    if min_val is not None:
        if array.min() < min_val:
            raise ValueError(
                f"The elements of {name} must be larger than {min_val}, but got minimum value {array.min()}"
            )
    if max_val is not None:
        if array.max() > max_val:
            raise ValueError(
                f"The elements of {name} must be smaller than {max_val}, but got maximum value {array.max()}"
            )
