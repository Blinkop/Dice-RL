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

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

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
        seed: int = None,
    ):
        super().__init__()

        self._torch_generator = torch.Generator()
        if seed is not None:
            self._torch_generator.manual_seed(seed)
        else:
            self._torch_generator.seed()

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]

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
                torch.nn.init.xavier_uniform_(param.data)
            except:
                torch.nn.init.normal_(param.data)

    def forward(self, states):
        if len(states.shape) != 2:
            raise ValueError(f"every state must be 1d vector")

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
        primal_form: bool = False,
        num_action_samples: Optional[int] = None,
        primal_regularizer: float = 0.0,
        dual_regularizer: float = 1.0,
        norm_regularizer: float = 0.0,
        nu_regularizer: float = 0.0,
        zeta_regularizer: float = 0.0,
        weight_by_gamma: bool = False,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()

        self._device = device

        self._nu_network = nu_network
        self._zeta_network = zeta_network
        self._lambda = nn.Parameter(data=torch.tensor(0.0).to(self._device))
        self._nu_optimizer = Adam(list(self._nu_network.parameters()), lr=nu_lr)
        self._zeta_optimizer = Adam(list(self._zeta_network.parameters()), lr=zeta_lr)
        self._lambda_optimizer = Adam([self._lambda], lr=lambda_lr)

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
            raise ValueError("Exponent for f must be greater than 1")
        f_star_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x) ** f_exponent / f_exponent
        self._f_star_fn = lambda x: torch.abs(x) ** f_star_exponent / f_star_exponent

    def nu_value_expectation(self, states: torch.Tensor, policy: Policy):
        batch_size = states.shape[0]

        if self._num_action_samples is None:
            action_weights = policy.action_dist(state=states)  # (B, NUM_ACTIONS)
        else:
            action_weights = (1 / self._num_action_samples) * torch.ones(
                (batch_size, self._num_action_samples)
            ).to(
                self._device
            )  # (B, A)

        # if weighting by policy.action_dist, then A = NUM_ACTIONS
        values = self._nu_network(states)  # (B, A)
        value_expectation = values * action_weights

        return value_expectation.sum(dim=1)  # (B,)

    def orthogonal_regularization_loss(self, network: ValueNetwork):
        reg = 0
        for layer in network.modules():
            if isinstance(layer, nn.Linear):
                prod = layer.weight @ layer.weight.T
                reg += torch.sum(
                    torch.square(prod * (1 - torch.eye(prod.shape[0]).to(self._device)))
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
        policy: Policy,
    ):
        nu_first_values = self.nu_value_expectation(
            states=first_state, policy=policy
        ).flatten()  # (B,)
        nu_current_values = (
            self._nu_network(current_state)
            .gather(1, current_action.view(-1, 1))
            .flatten()
        )  # (B,)
        nu_next_values = self.nu_value_expectation(
            states=next_state, policy=policy
        ).flatten()  # (B,)
        zeta_current_values = (
            self._zeta_network(current_state)
            .gather(1, current_action.view(-1, 1))
            .flatten()
        )  # (B,)

        # discount = self._gamma ** step_num # (B,)
        discount = self._gamma

        bellman_residuals = (
            discount * nu_next_values
            - nu_current_values
            - self._norm_regularizer * self._lambda
        )

        if not self._zero_reward:
            bellman_residuals += rewards

        zeta_loss = -zeta_current_values * bellman_residuals
        nu_loss = (1 - self._gamma) * nu_first_values
        lambda_loss = self._norm_regularizer * self._lambda

        if self._primal_form:
            nu_loss += self._f_star_fn(bellman_residuals)
            lambda_loss += self._f_star_fn(bellman_residuals)
        else:
            nu_loss += zeta_current_values * bellman_residuals
            lambda_loss = (
                lambda_loss
                - self._norm_regularizer * self._lambda * zeta_current_values
            )

        nu_loss += self._primal_regularizer * self._f_fn(nu_current_values)
        zeta_loss += self._dual_regularizer * self._f_fn(zeta_current_values)

        if self._weight_by_gamma:
            weights = self._gamma**step_num
            weights /= 1e-6 + weights.mean()
            nu_loss *= weights
            zeta_loss *= weights

        return nu_loss.mean(), zeta_loss.mean(), lambda_loss.mean()

    def train_batch(self, batch, policy: Policy):
        self._nu_network.train()
        self._zeta_network.train()

        (
            first_state,  # (B, S)
            current_state,  # (B, S)
            current_action,  # (B,)
            next_state,  # (B, S)
            rewards,  # (B,)
            step_num,  # (B,)
        ) = batch

        nu_loss, zeta_loss, lambda_loss = self.train_loss(
            first_state=first_state,
            current_state=current_state,
            current_action=current_action,
            next_state=next_state,
            rewards=rewards,
            step_num=step_num,
            policy=policy,
        )

        nu_loss += self._nu_regularizer * self.orthogonal_regularization_loss(
            network=self._nu_network
        )
        zeta_loss += self._zeta_regularizer * self.orthogonal_regularization_loss(
            network=self._zeta_network
        )

        self._nu_optimizer.zero_grad()
        self._zeta_optimizer.zero_grad()
        self._lambda_optimizer.zero_grad()

        nu_loss.backward(retain_graph=True)
        zeta_loss.backward(retain_graph=True)
        lambda_loss.backward()

        self._nu_optimizer.step()
        self._zeta_optimizer.step()
        self._lambda_optimizer.step()

        return nu_loss.item(), zeta_loss.item(), lambda_loss.item()

    def estimate_average_reward(
        self,
        states: torch.Tensor,  # (B, S)
        actions: torch.Tensor,  # (B,)
        rewards: torch.Tensor,  # (B,)
        print_zeta=False,
    ):
        self._nu_network.eval()
        self._zeta_network.eval()

        with torch.no_grad():
            weights = (
                self._zeta_network(states).gather(1, actions.view(-1, 1)).flatten()
            )
            if print_zeta:
                print(weights.mean(), weights.min(), weights.max())
            result = torch.mean(weights * rewards).detach().cpu()

        return result.item()
