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

from typing import Type

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


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
        multihead_output: bool = False,
        output_activation: Type[nn.Module] = None,
        seed: int = None
    ):
        super().__init__()

        self._multihead_output = multihead_output

        self._torch_generator = torch.Generator()
        if seed is not None:
            self._torch_generator.manual_seed(seed)
        else:
            self._torch_generator.seed()

        input_size = state_dim if self._multihead_output else state_dim + action_dim
        output_size = action_dim if self._multihead_output else 1

        layers = [
            nn.Linear(input_size, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_size))
        if output_activation is not None:
            layers.append(output_activation())

        self._network = nn.Sequential(*layers)

        self.init_weights()

    @property
    def multihead_output(self):
        return self._multihead_output

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

    def forward(self, states, actions=None):
        if len(states.shape) != 2:
            raise ValueError(f'every state must be 1d vector')

        if not self._multihead_output and actions is None:
            raise ValueError(f'action embeddings must be passed when multihead=False')
        
        if not self._multihead_output and len(actions.shape) != 2:
            raise ValueError(f'every action must be 1d vector')
        
        if self._multihead_output:
            input_batch = states
        else:
            input_batch = torch.concat([states, actions], dim=1)

        return self._network(input_batch)


class NeuralDice(object):
    def __init__(
        self,
        action_embeddigns: torch.Tensor,
        nu_network: ValueNetwork,
        zeta_network: ValueNetwork,
        nu_lr: float,
        zeta_lr: float,
        lambda_lr: float,
        lr_schedule: bool = False,
        gamma: float = 0.99,
        zero_reward: bool = False,
        f_exponent: float = 1.5,
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

        self._action_embs = action_embeddigns.to(self._device)

        self._nu_network = nu_network
        self._zeta_network = zeta_network
        self._lambda = nn.Parameter(data=torch.tensor(0.0).to(self._device))
        self._lr_schedule = lr_schedule
        self._nu_optimizer = Adam(list(self._nu_network.parameters()), lr=nu_lr)
        self._zeta_optimizer = Adam(
            list(self._zeta_network.parameters()), lr=zeta_lr, maximize=True)
        self._lambda_optimizer = Adam([self._lambda], lr=lambda_lr)

        self._nu_lr_scheduler = StepLR(
            optimizer=self._nu_optimizer,
            step_size=1000,
            gamma=0.1**(1/200),
        )
        self._zeta_lr_scheduler = StepLR(
            optimizer=self._zeta_optimizer,
            step_size=1000,
            gamma=0.1**(1/200),
        )
        self._lambda_lr_scheduler = StepLR(
            optimizer=self._lambda_optimizer,
            step_size=1000,
            gamma=0.1**(1/200),
        )

        self._multihead_output = self._nu_network.multihead_output

        self._zero_reward = zero_reward
        self._weight_by_gamma = weight_by_gamma

        self._nu_regularizer = nu_regularizer
        self._zeta_regularizer = zeta_regularizer
        self._primal_regularizer = primal_regularizer
        self._dual_regularizer = dual_regularizer
        self._norm_regularizer = norm_regularizer

        self._gamma = gamma

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1')
        f_star_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x) ** f_exponent / f_exponent
        self._f_star_fn = lambda x: torch.abs(x) ** f_star_exponent / f_star_exponent


    def nu_value_expectation(
        self,
        states: torch.Tensor, # (B, S)
        sampled_actions: torch.Tensor, # (B, A)
    ):
        batch_size = states.shape[0]
        num_sampled_actions = sampled_actions.shape[1]

        action_weights = (1 / num_sampled_actions) *\
            torch.ones((batch_size, num_sampled_actions)).to(self._device) # (B, A)
        
        if self._multihead_output:
            values = self._nu_network(states).gather(1, sampled_actions) # (B, A)
        else:
            states = states.repeat(1, num_sampled_actions)\
                .reshape(batch_size * num_sampled_actions, -1) # (B*A, S)
            actions = self._action_embs[sampled_actions.view(-1)] # (B*A, E)
            
            values = self._nu_network(states, actions).reshape(batch_size, -1) # (B, A)
        
        
        value_expectation = values * action_weights

        return value_expectation.sum(dim=1) # (B,)

    def train_loss(
        self,
        first_state: torch.Tensor, # (B, S)
        first_sampled_actions: torch.Tensor, # (B, A)
        current_state: torch.Tensor, # (B, S)
        current_action: torch.Tensor, # (B,)
        next_state: torch.Tensor, # (B, S)
        next_sampled_actions: torch.Tensor, # (B, A)
        rewards: torch.Tensor, # (B,)
        step_num: torch.Tensor, # (B,)
        has_next: torch.Tensor, # (B,)
    ):
        if self._multihead_output:
            nu_current_values = self._nu_network(current_state)\
                .gather(1, current_action.view(-1, 1)).flatten() # (B,)
            zeta_current_values = self._zeta_network(current_state)\
                .gather(1, current_action.view(-1, 1)).flatten() # (B,)
        else:
            nu_current_values = self._nu_network(
                current_state, self._action_embs[current_action]
            ).flatten() # (B,)
            zeta_current_values = self._zeta_network(
                current_state, self._action_embs[current_action]
            ).flatten() # (B,)

        nu_first_values = self.nu_value_expectation(
            states=first_state,
            sampled_actions=first_sampled_actions
        ).flatten() # (B,)
        nu_next_values = self.nu_value_expectation(
            states=next_state,
            sampled_actions=next_sampled_actions
        ).flatten() # (B,)
        
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

    def train_batch(self, batch):
        self._nu_network.train()
        self._zeta_network.train()

        (
            first_state,
            first_sampled_actions,
            current_state,
            current_action,
            next_state,
            next_sampled_actions,
            rewards,
            step_num,
            has_next
        ) = batch

        loss = self.train_loss(
            first_state=first_state,
            first_sampled_actions=first_sampled_actions,
            current_state=current_state,
            current_action=current_action,
            next_state=next_state,
            next_sampled_actions=next_sampled_actions,
            rewards=rewards,
            step_num=step_num,
            has_next=has_next
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

        if self._lr_schedule:
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
            if self._multihead_output:
                weights = self._zeta_network(states).gather(1, actions.view(-1, 1)).flatten()
            else:
                weights = self._zeta_network(states, self._action_embs[actions]).flatten()
            
            result = torch.sum(weights * rewards).detach().cpu()

        return result.item()
