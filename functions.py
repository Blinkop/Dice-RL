from enum import Enum

import torch
import torch.nn as nn

from sklearn.utils import check_scalar


class DiceFunctions(Enum):
    P_2                 = lambda x: x ** 2 / 2
    P_3                 = lambda x: torch.abs(x) ** 3 / 3
    DUAL_DICE_P_3_2     = lambda x: torch.abs(x) ** 1.5 / 1.5
    SCOPE_RL_P_2        = lambda x: x ** 2
    CHI_SQUARED         = lambda x: (x - 1) ** 2 / 2
    NEYMAN_CHI_SQUARED  = lambda x: (x - 1) ** 2
    PEARSON_CHI_SQUARED = lambda x: (x - 1) ** 2 / x
    KL_DIVERGENCE       = lambda x: x * torch.log(torch.abs(x))


class StateActionNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int,
        seed: int = None
    ) -> None:
        super().__init__()

        check_scalar(num_layers, name='num_layers', target_type=int, min_val=2)

        self._torch_generator = torch.Generator()
        if seed is not None:
            self._torch_generator.manual_seed(seed)
        else:
            self._torch_generator.seed()

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

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        raise NotImplementedError()


class DiscreteSANetwork(StateActionNetwork):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int,
        action_emb: torch.Tensor = None,
        seed: int = None,
        device: str = 'cpu'
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            seed=seed
        )

        if action_emb is None:
            action_emb = torch.eye(action_dim)

        self._action_emb = action_emb.to(device)

        layers = [
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self._network = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        s_a = torch.cat([
            states,
            self._action_emb[actions]
        ], dim=1)

        return self._network(s_a).flatten()


class MultiheadDiscreteSANetwork(StateActionNetwork):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int,
        seed: int = None
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            seed=seed
        )

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, action_dim))

        self._network = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        values = self._network(states)

        return values.gather(1, actions.view(-1, 1)).flatten()
