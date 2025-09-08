from typing import Optional, Callable, List
from abc import ABC
from pathlib import Path

import json

from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sklearn.utils import check_scalar

from functions import StateActionNetwork
from functions import DiceFunctions
from utils import DiceDatasetWrapper, check_array, get_lambda_code, custom_collate

import seaborn as sns
import matplotlib.pyplot as plt


class Dice(ABC):
    def __init__(
        self,
        q_function: StateActionNetwork,
        w_function: StateActionNetwork,
        gamma: float = 0.99,
        q_lr: float = 1e-4,
        w_lr: float = 1e-4,
        lambda_lr: float = 1e-4,
        f1_function: Callable[[torch.Tensor], torch.Tensor] = DiceFunctions.SCOPE_RL_P_2,
        f2_function: Callable[[torch.Tensor], torch.Tensor] = DiceFunctions.CHI_SQUARED,
        method_name: str = 'dual_dice',
        alpha_q: Optional[float] = None,
        alpha_w: Optional[float] = None,
        alpha_r: Optional[float] = None,
        w_positivity: Optional[bool] = None,
        enable_lambda: Optional[bool] = None,
        seed: int = None,
        device: str = 'cpu'
    ) -> None:
        super().__init__()

        check_scalar(
            gamma, name='gamma', target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(q_lr, name='q_lr', target_type=float, min_val=0.0)
        check_scalar(w_lr, name='w_lr', target_type=float, min_val=0.0)
        check_scalar(lambda_lr, name='lambda_lr', target_type=float, min_val=0.0)

        self._seed = seed
        self._device = device

        self._torch_generator = torch.Generator()
        if self._seed is not None:
            self._torch_generator.manual_seed(self._seed)
        else:
            self._torch_generator.seed()

        if method_name not in [
            'dual_dice',
            'gen_dice',
            'gradient_dice',
            'mql',
            'mwl',
            'best_dice',
            'custom'
        ]:
            raise ValueError(f'unknown method name "{method_name}"')
        
        if method_name == 'custom':
            self._alpha_q = alpha_q
            self._alpha_w = alpha_w
            self._alpha_r = alpha_r
            self._w_positivity = w_positivity
            self._enable_lambda = enable_lambda
        else:
            self._alpha_q = float(method_name in ['gen_dice', 'gradient_dice'])
            self._alpha_w = float(method_name in ['dual_dice', 'best_dice'])
            self._alpha_r = method_name in ['mql', 'best_dice']
            self._w_positivity = method_name in ['gen_dice', 'best_dice']
            self._enable_lambda = method_name in ['gen_dice', 'gradient_dice', 'best_dice']

        self._q_function = q_function.to(self._device)
        self._w_function = w_function.to(self._device)

        self._q_lr = q_lr
        self._w_lr = w_lr
        self._lambda_lr = lambda_lr

        self._gamma = gamma
        self._f1_function = f1_function
        self._f2_function = f2_function

        self._q_optimizer = Adam(self._q_function.parameters(), lr=self._q_lr)
        self._w_optimizer = Adam(self._w_function.parameters(), lr=self._w_lr, maximize=True)

        if self._enable_lambda:
            self._lambda = torch.ones((1,), device=self._device, requires_grad=True)
            self._lambda_optimizer = Adam([self._lambda], lr=self._lambda_lr)
        else:
            self._lambda = torch.zeros((1,), device=self._device)

        self._reset_experiment_data()


    def _reset_experiment_data(self):
        self._experiment_data = {
            'initial_loss' : [],
            'td_loss' : [],
            'q_reg_loss' : [],
            'w_reg_loss' : [],
            'loss' : [],
            'value_per_step' : []
        }

    def _write_experiment_data(self, experiment_name: str):
        exp_folder = Path(f"experiments/{experiment_name}")
        exp_folder.mkdir(parents=True, exist_ok=True)

        params = {
            'q_lr' : self._q_lr,
            'w_lr' : self._w_lr,
            'lambda_lr' : self._lambda_lr,
            'gamma' : self._gamma,
            'alpha_q' : self._alpha_q,
            'alpha_w' : self._alpha_w,
            'alpha_r' : self._alpha_r,
            'w_positivity' : self._w_positivity,
            'enable_lambda' : self._enable_lambda,
            'f1_function' : get_lambda_code(self._f1_function),
            'f2_function' : get_lambda_code(self._f2_function),
            'seed' : self._seed
        }

        with open(exp_folder / "parameters.json", 'w') as f:
            json.dump(params, f, indent="\t")

        sns.set_theme()
        plt.title("loss")
        plt.plot(self._experiment_data['initial_loss'], alpha=0.7, label='initial q')
        plt.plot(self._experiment_data['td_loss'], alpha=0.7, label='td')
        plt.plot(self._experiment_data['q_reg_loss'], alpha=0.7, label='q reg')
        plt.plot(self._experiment_data['w_reg_loss'], alpha=0.7, label='w reg')
        plt.plot(self._experiment_data['loss'], alpha=0.7, label='loss')
        plt.legend()
        plt.savefig(exp_folder / "losses.png", bbox_inches="tight")
        plt.close()

        sns.set_theme()
        plt.title(f"values")
        plt.plot(self._experiment_data['value_per_step'])
        plt.savefig(exp_folder / "values.png", bbox_inches="tight")
        plt.close()

    def _report_loss(
        self,
        initial: float,
        td: float,
        q_reg: float,
        w_reg: float,
        loss: float
    ):
        self._experiment_data['initial_loss'].append(initial)
        self._experiment_data['td_loss'].append(td)
        self._experiment_data['q_reg_loss'].append(q_reg)
        self._experiment_data['w_reg_loss'].append(w_reg)
        self._experiment_data['loss'].append(loss)

    def _report_value(self, value: float):
        self._experiment_data['value_per_step'].append(value)


    def objective_function(
        self,
        first_state: torch.Tensor,
        first_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor
    ):
        initial_value = (1 - self._gamma) * self._q_function(
            first_state, first_action
        ).mean() + self._lambda

        q = self._q_function(state, action)
        w = self._w_function(state, action)

        if self._w_positivity:
            w = w**2

        td_loss = (
            w * (
                self._alpha_r * reward
                + self._gamma * self._q_function(next_state, next_action)
                - q
                - self._lambda
            )
        ).mean()

        q_regularization = self._alpha_q * self._f1_function(q).mean()
        w_regularization = self._alpha_w * self._f2_function(w).mean()

        return initial_value, td_loss, q_regularization, w_regularization

    
    def fit(
        self,
        state: List[np.ndarray],
        action: List[np.ndarray],
        reward: List[np.ndarray],
        target_action: List[np.ndarray],
        num_steps: int,
        batch_size: int = 1024,
        eval_iter: int = 100,
        num_workers: int = 4,
        result_folder: str = None
    ):
        check_scalar(num_steps, name='num_steps', target_type=int, min_val=1)
        check_scalar(batch_size, name='batch_size', target_type=int, min_val=1)
        check_scalar(num_workers, name='num_workers', target_type=int, min_val=1)

        if not (
            len(state)
            == len(action)
            == len(reward)
            == len(target_action)
        ):
            raise ValueError('number of trajectories mismatch')

        for i in range(len(state)):
            check_array(state[i], name=f'state[{i}]', expected_dim=2)
            check_array(action[i], name=f'action[{i}]', expected_dim=1)
            check_array(reward[i], name=f'reward[{i}]', expected_dim=1)
            check_array(target_action[i], name=f'target_action[{i}]', expected_dim=1)

            if not (
                state[i].shape[0]
                == action[i].shape[0]
                == reward[i].shape[0]
                == target_action[i].shape[0]
            ):
                raise ValueError(f'trajectory length mismatch at index {i}')

        if batch_size % num_workers > 0:
            raise ValueError(f'batch_size % num_workers != 0')

        self._reset_experiment_data()

        single_batch_size = int(batch_size / num_workers)

        dataset = DiceDatasetWrapper(
            states=state,
            actions=action,
            rewards=reward,
            target_actions=target_action,
            batch_size=single_batch_size
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=num_workers,
            num_workers=num_workers,
            prefetch_factor=None,
            pin_memory=True,
            collate_fn=custom_collate,
            persistent_workers=False,
            generator=self._torch_generator
        )

        tqdm_iterator = tqdm(loader, total=num_steps)
        for i, batch in enumerate(tqdm_iterator):
            if i >= num_steps:
                break

            initial_, td_, q_, w_ = self.objective_function(
                first_state=batch[0],
                first_action=batch[1],
                state=batch[2],
                action=batch[3],
                reward=batch[4],
                next_state=batch[5],
                next_action=batch[6]
            )
            loss = initial_ + td_ + q_ - w_

            self._q_optimizer.zero_grad()
            self._w_optimizer.zero_grad()
            if self._enable_lambda:
                self._lambda_optimizer.zero_grad()

            loss.backward()

            clip_grad_norm_(self._q_function.parameters(), max_norm=10.0)
            clip_grad_norm_(self._w_function.parameters(), max_norm=10.0)
            if self._enable_lambda:
                clip_grad_norm_(self._lambda, max_norm=10.0)

            self._q_optimizer.step()
            self._w_optimizer.step()
            if self._enable_lambda:
                self._lambda_optimizer.step()

            if i % eval_iter == 0 or i == (num_steps - 1):
                value = self.predict_per_step_reward(
                    state=state,
                    action=action,
                    reward=reward
                )

                if result_folder is not None:
                    self._report_value(value)

            tqdm_iterator.set_description(
                f'loss: {loss.item():.4f}; value: {value:.4f}'
            )

            if result_folder is not None:
                self._report_loss(
                    initial_.item(), td_.item(),
                    q_.item(), w_.item(), loss.item()
                )

        if result_folder is not None:
            self._write_experiment_data(result_folder)


    @torch.no_grad()
    def predict_weights(self, state: np.ndarray, action: np.ndarray):
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=1)

        if state.shape[0] != action.shape[0]:
            raise ValueError("len(states) != len(actions)")
        
        state = torch.tensor(state, dtype=torch.float, device=self._device)
        action = torch.tensor(action, dtype=torch.long, device=self._device)

        w = self._w_function(state, action)

        if self._w_positivity:
            w = w**2

        return w.to('cpu').detach().numpy()
    
    def predict_per_step_reward(
        self,
        state: List[np.ndarray],
        action: List[np.ndarray],
        reward: List[np.ndarray]
    ):
        w = self.predict_weights(
            np.concatenate(state),
            np.concatenate(action)
        )

        r = np.concatenate(reward)

        return (w * r).mean().item()

    def predict_per_traj_reward(
        self,
        state: List[np.ndarray],
        action: List[np.ndarray],
        reward: List[np.ndarray]
    ):
        per_step_rewards = self.predict_per_step_reward(
            state=state,
            action=action,
            reward=reward
        )

        total_steps = sum([s.shape[0] for s in state])

        return (total_steps / len(state)) * per_step_rewards
