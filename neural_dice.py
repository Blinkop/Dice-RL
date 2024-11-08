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

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam

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
        squared_output: bool = True
    ):
        super().__init__()

        layers = [
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        if squared_output:
            layers.append(SquaredActivation())

        self._network = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                torch.nn.init.normal_(param.data)

    def forward(self, states, actions):
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f'There must be the same batch size of states and actions')
        
        if len(states.shape) != 2 or len(actions.shape) != 2:
            raise ValueError(f'every state and action must be 1d vector')

        state_action = torch.concat([states, actions], dim=1)

        return self._network(state_action)


class SingleVariable(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()

        self.variable = nn.Parameter(torch.Tensor([init_value]).requires_grad_())

    def forward(self):
        return self.variable



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
        primal_form: bool = False,
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
        # self._lambda = nn.Parameter(data=torch.Tensor([0.0]).to(self._device))
        self._nu_optimizer = Adam(list(self._nu_network.parameters()), lr=nu_lr)
        self._zeta_optimizer = Adam(list(self._zeta_network.parameters()), lr=zeta_lr)
        # self._lambda_optimizer = Adam([self._lambda], lr=lambda_lr)

        self._zero_reward = zero_reward
        self._primal_form = primal_form
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


    def value_expectation(
        self,
        states: torch.Tensor,
        policy: Policy
    ):
        batch_size = states.shape[0]

        if self._num_action_samples is None:
            action_weights = policy.action_dist(state=states) # (B, NUM_ACTIONS)
            actions = torch.tile(
                torch.arange(self._num_actions),
                dims=(batch_size, 1)
            ).to(self._device) # (B, NUM_ACTIONS)
        else:
            action_weights = (1 / self._num_action_samples) *\
                torch.ones((batch_size, self._num_action_samples)).to(self._device) # (B, A)
            actions = torch.concat([
                policy.select_action(state=states).reshape(-1, 1)
                for _ in range(self._num_action_samples)
            ], dim=1) # (B, A)

        # if weighting by policy.action_dist, then A = NUM_ACTIONS
        actions_flatten = actions.flatten() # (B, A) -> (B*A)
        actions_flatten = F.one_hot(
            actions_flatten, self._num_actions
        ).to(self._device) # (B*A) -> (B*A, NUM_ACTIONS)
        states_flatten = states.repeat(1, action_weights.shape[1])\
            .reshape(-1, states.shape[1])\
            .to(self._device) # (B, S) -> (B*A, S)

        values = self._nu_network(states_flatten, actions_flatten)\
            .reshape(batch_size, -1) # (B, A)

        value_expectation = values * action_weights

        return value_expectation.sum(dim=1) # (B,)

    def orthogonal_regularization_loss(self, network: ValueNetwork):
        reg = 0
        for layer in network.modules():
            if isinstance(layer, nn.Linear):
                prod = layer.weight @ layer.weight.T
                reg += torch.sum(
                    torch.square(
                        prod * (1 - torch.eye(prod.shape[0]).to(self._device))
                    )
                )

        return reg


    def train_loss(
        self,
        first_state: torch.Tensor,
        current_state: torch.Tensor,
        current_action: torch.Tensor,
        next_state: torch.Tensor,
        rewards: torch.Tensor,
        step_num: torch.Tensor,
        policy: Policy
    ):
        current_action = F.one_hot(current_action, self._num_actions).to(self._device)

        nu_first_values = self.value_expectation(states=first_state, policy=policy).flatten() # (B,)
        nu_current_values = self._nu_network(current_state, current_action).flatten() # (B,)
        nu_next_values = self.value_expectation(states=next_state, policy=policy).flatten() # (B,)
        zeta_current_values = self._zeta_network(current_state, current_action).flatten() # (B,)

        # discount = self._gamma ** step_num # (B,)
        discount = self._gamma

        bellman_residuals = discount * nu_next_values\
            - nu_current_values# - self._norm_regularizer * self._lambda
        
        if not self._zero_reward:
            bellman_residuals += rewards

        zeta_loss = -zeta_current_values * bellman_residuals
        nu_loss = (1 - self._gamma) * nu_first_values
        # lambda_loss = self._norm_regularizer * self._lambda

        if self._primal_form:
            nu_loss += self._f_star_fn(bellman_residuals)
            # lambda_loss += self._f_star_fn(bellman_residuals)
        else:
            nu_loss += zeta_current_values * bellman_residuals
            # lambda_loss -= self._norm_regularizer * zeta_current_values * self._lambda

        nu_loss += self._primal_regularizer * self._f_fn(nu_current_values)
        zeta_loss += self._dual_regularizer * self._f_fn(zeta_current_values)

        if self._weight_by_gamma:
            weights = self._gamma ** step_num
            weights /= 1e-6 + weights.mean()
            nu_loss *= weights
            zeta_loss *= weights

        return nu_loss.mean(), zeta_loss.mean()

    def train_batch(self, batch, policy: Policy):
        self._nu_network.train()
        self._zeta_network.train()

        (
            first_state, # (B, S)
            current_state, # (B, S)
            current_action, # (B,)
            next_state, # (B, S)
            rewards, # (B,)
            step_num # (B,)
        ) = batch

        nu_loss, zeta_loss = self.train_loss(
            first_state=first_state,
            current_state=current_state,
            current_action=current_action,
            next_state=next_state,
            rewards=rewards,
            step_num=step_num,
            policy=policy
        )

        nu_loss += self._nu_regularizer * self.orthogonal_regularization_loss(
            network=self._nu_network)
        zeta_loss += self._zeta_regularizer * self.orthogonal_regularization_loss(
            network=self._zeta_network)

        self._nu_optimizer.zero_grad()
        self._zeta_optimizer.zero_grad()

        nu_loss.backward(retain_graph=True)
        zeta_loss.backward()

        self._nu_optimizer.step()
        self._zeta_optimizer.step()

        return nu_loss.item(), zeta_loss.item()

    def estimate_average_reward(
        self,
        states: torch.Tensor,  # (B, S)
        actions: torch.Tensor, # (B,)
        rewards: torch.Tensor, # (B,)
    ):
        self._nu_network.eval()
        self._zeta_network.eval()

        with torch.no_grad():
            weights = self._zeta_network(states, actions).flatten()
            result = torch.sum(weights * rewards).detach().cpu()

        return result.item()
