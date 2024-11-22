# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from policy import Policy


class SquaredActivation(nn.Module):
    """Returns square of input"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        output_activation: Type[nn.Module] = None,
        seed: int = None
    ):
        super().__init__()

        self._torch_generator = torch.Generator()
        if seed is not None:
            self._torch_generator.manual_seed(seed)
        else:
            self._torch_generator.seed()

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, action_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self._network = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(
                    param.data, generator=self._torch_generator
                )
            except:
                torch.nn.init.normal_(
                    param.data, generator=self._torch_generator
                )

    def forward(self, states):
        if len(states.shape) != 2:
            raise ValueError(f'every state must be 1d vector')

        return self._network(states)


class NeuralDice(object):
    def __init__(
        self,
        nu_network: ValueNetwork,
        zeta_network: ValueNetwork,
        nu_lr: float,
        zeta_lr: float,
        lambda_lr: float,
        num_actions: int,
        gamma: float,
        zero_reward: bool = False,
        f_exponent: float = 1.5,
        num_action_samples: Optional[int] = None,
        primal_regularizer: float = 0.0,
        dual_regularizer: float = 1.0,
        norm_regularizer: float = 0.0,
        nu_regularizer: float = 0.0,
        zeta_regularizer: float = 0.0,
        weight_by_gamma: bool = False,
        device: torch.device = torch.device('cuda:0')
    ):
        super().__init__()

        self._device = device

        self._nu_network = nu_network
        self._zeta_network = zeta_network
        self._lambda = nn.Parameter(data=torch.tensor(0.0).to(self._device))
        self._nu_optimizer = Adam(list(self._nu_network.parameters()), lr=nu_lr)
        self._zeta_optimizer = Adam(
            list(self._zeta_network.parameters()), lr=zeta_lr, maximize=True)
        self._lambda_optimizer = Adam([self._lambda], lr=lambda_lr)

        self._nu_lr_scheduler = CosineAnnealingLR(
            optimizer=self._nu_optimizer,
            T_max=1000
        )
        self._zeta_lr_scheduler = CosineAnnealingLR(
            optimizer=self._zeta_optimizer,
            T_max=1000
        )
        self._lambda_lr_scheduler = CosineAnnealingLR(
            optimizer=self._lambda_optimizer,
            T_max=1000
        )

        self._zero_reward = zero_reward
        self._weight_by_gamma = weight_by_gamma

        self._nu_regularizer = nu_regularizer
        self._zeta_regularizer = zeta_regularizer
        self._primal_regularizer = primal_regularizer
        self._dual_regularizer = dual_regularizer
        self._norm_regularizer = norm_regularizer

        self._gamma = gamma

        self._num_actions = num_actions
        self._num_action_samples = num_action_samples

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1')
        f_star_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x) ** f_exponent / f_exponent
        self._f_star_fn = lambda x: torch.abs(x) ** f_star_exponent / f_star_exponent


    def nu_value_expectation(
        self,
        states: torch.Tensor,
        policy: Policy,
        inputs: List
    ):
        batch_size = states.shape[0]

        if self._num_action_samples is None:
            action_weights = policy.action_dist(state=inputs) # (B, NUM_ACTIONS)
            values = self._nu_network(states) # (B, NUM_ACTIONS)
        else:
            action_weights = (1 / self._num_action_samples) *\
                torch.ones((batch_size, self._num_action_samples)).to(self._device) # (B, A)
            actions = torch.concat([
                policy.select_action(state=inputs).reshape(-1, 1)
                for _ in range(self._num_action_samples)
            ], dim=1) # (B, A)
            values = self._nu_network(states).gather(1, actions) # (B, A)
        
        value_expectation = values * action_weights

        return value_expectation.sum(dim=1) # (B,)

    def train_loss(
        self,
        first_state: torch.Tensor,
        first_policy_inputs: List,
        current_state: torch.Tensor,
        current_action: torch.Tensor,
        next_state: torch.Tensor,
        next_policy_inputs: List,
        rewards: torch.Tensor,
        step_num: torch.Tensor,
        has_next: torch.Tensor,
        policy: Policy
    ):
        nu_first_values = self.nu_value_expectation(
            states=first_state, policy=policy, inputs=first_policy_inputs
        ).flatten() # (B,)
        nu_current_values = self._nu_network(current_state)\
            .gather(1, current_action.view(-1, 1))\
            .flatten() # (B,)
        nu_next_values = self.nu_value_expectation(
            states=next_state, policy=policy, inputs=next_policy_inputs
        ).flatten() # (B,)
        zeta_current_values = self._zeta_network(current_state)\
            .gather(1, current_action.view(-1, 1))\
            .flatten() # (B,)

        discount = self._gamma * has_next

        bellman_residuals = (
            discount * nu_next_values
            - nu_current_values
            - self._norm_regularizer * self._lambda
        ) # (B,)
        
        if not self._zero_reward:
            bellman_residuals = bellman_residuals + rewards

        loss = (
            (1 - self._gamma) * nu_first_values
            + zeta_current_values * bellman_residuals
            + self._norm_regularizer * self._lambda
            - self._f_fn(zeta_current_values)
            + self._primal_regularizer * self._f_fn(nu_current_values)
            - self._dual_regularizer * self._f_fn(zeta_current_values)
        )

        if self._weight_by_gamma:
            weights = self._gamma ** step_num
            weights = weights / (weights.mean() + 1e-6)
            loss = loss * weights

        return loss.mean()

    def train_batch(self, batch, policy: Policy):
        self._nu_network.train()
        self._zeta_network.train()

        (
            first_state, # (B, S)
            first_policy_inputs, # len() = B
            current_state, # (B, S)
            current_action, # (B,)
            next_state, # (B, S)
            next_policy_inputs, # len() = B
            rewards, # (B,)
            step_num, # (B,)
            has_next # (B,)
        ) = batch

        loss = self.train_loss(
            first_state=first_state,
            first_policy_inputs=first_policy_inputs,
            current_state=current_state,
            current_action=current_action,
            next_state=next_state,
            next_policy_inputs=next_policy_inputs,
            rewards=rewards,
            step_num=step_num,
            has_next=has_next,
            policy=policy
        )

        self._nu_optimizer.zero_grad()
        self._zeta_optimizer.zero_grad()
        self._lambda_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._nu_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            self._zeta_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            [self._lambda], max_norm=1.0)

        self._nu_optimizer.step()
        self._zeta_optimizer.step()
        self._lambda_optimizer.step()

        self._nu_lr_scheduler.step()
        self._zeta_lr_scheduler.step()
        self._lambda_lr_scheduler.step()

        return loss.item()

    def estimate_average_reward(
        self,
        states: torch.Tensor,  # (B, S)
        actions: torch.Tensor, # (B,)
        rewards: torch.Tensor, # (B,)
    ):
        self._nu_network.eval()
        self._zeta_network.eval()

        with torch.no_grad():
            weights = self._zeta_network(states)\
                .gather(1, actions.view(-1 ,1))\
                .flatten()
            result = torch.sum(weights * rewards).detach().cpu()

        return result.item()
