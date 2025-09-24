import argparse
from typing import List, Callable
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import RandomSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact

from dice import Dice
from functions import (
    DiscreteSANetwork, MultiheadDiscreteSANetwork,
    DiceFunctions
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-na",
        "--num_actions",
        help="number of actions",
        type=str,
    )
    parser.add_argument(
        "-sp",
        "--states_path",
        help="states file",
        type=str,
    )
    parser.add_argument(
        "-ap",
        "--actions_path",
        help="behavior policy actions file",
        type=str,
    )
    parser.add_argument(
        "-rp",
        "--rewards_path",
        help="rewards file",
        type=str,
    )
    parser.add_argument(
        "-s0",
        "--s0_target_actions_path",
        help="target policy actions file",
        type=str,
    )
    parser.add_argument(
        "-s1",
        "--s1_target_actions_path",
        help="target policy actions file",
        type=str,
    )
    parser.add_argument(
        "-s2",
        "--s2_target_actions_path",
        help="target policy actions file",
        type=str,
    )
    parser.add_argument(
        "-aep",
        "--action_emb_path",
        help="action embeding file",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="device to use",
        type=str,
    )

    return parser.parse_args()



class Objective:

    @dataclass
    class Parameters:
        
        num_layers: int
        hidden_dim: int
        is_multihead: bool

        gamma: float
        lr: float
        f1_func: Callable[[torch.Tensor], torch.Tensor]
        f2_func: Callable[[torch.Tensor], torch.Tensor]

        method_name: str
        num_steps: int
        batch_size: int
        eval_iter: int


    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[np.ndarray],
        s0_actions: List[np.ndarray],
        s1_actions: List[np.ndarray],
        s2_actions: List[np.ndarray],
        action_embs: torch.Tensor,
        num_actions: int,
        artifact_store: FileSystemArtifactStore,
        tmp_folder: str,
        device: str = "cpu"
    ):
        self._normalized_return = {
            "sasrec_0" : 0.5286067306600946,
            "sasrec_1" : 0.45137425709353585,
            "sasrec_2" : 0.15643613788010788,
        }
        
        self._artifact_store = artifact_store
        self._tmp_folder = tmp_folder
        self._device = device
        self._seeds = [89219, 1237, 113001]

        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._s0_actions = s0_actions
        self._s1_actions = s1_actions
        self._s2_actions = s2_actions
        self._action_embs = action_embs

        self._state_dim = self._states[0].shape[1]
        self._action_dim = self._action_embs.shape[1]
        self._num_actions = num_actions

    def _fit_instance(
        self,
        params: Parameters,
        target_actions: List[np.ndarray],
        target_policy_name: str,
        seed: int
    ):
        if params.is_multihead:
            args = {
                'state_dim' : self._state_dim,
                'hidden_dim' : params.hidden_dim,
                'action_dim' : self._num_actions,
                'num_layers' : params.num_layers,
                'seed' : seed
            }

            q_func = MultiheadDiscreteSANetwork(**args)
            w_func = MultiheadDiscreteSANetwork(**args)
        else:
            args = {
                'state_dim' : self._state_dim,
                'hidden_dim' : params.hidden_dim,
                'action_dim' : self._action_dim,
                'num_layers' : params.num_layers,
                'action_emb' : self._action_embs,
                'seed' : seed,
                'device' : self._device
            }

            q_func = DiscreteSANetwork(**args)
            w_func = DiscreteSANetwork(**args)

        dice = Dice(
            q_function=q_func,
            w_function=w_func,
            gamma=params.gamma,
            q_lr=params.lr,
            w_lr=params.lr,
            lambda_lr=params.lr,
            f1_function=params.f1_func,
            f2_function=params.f2_func,
            method_name=params.method_name,
            seed=seed,
            device=self._device
        )

        dice.fit(
            state=self._states,
            action=self._actions,
            reward=self._rewards,
            target_action=target_actions,
            num_steps=params.num_steps,
            batch_size=params.batch_size,
            eval_iter=params.eval_iter,
            num_workers=8,
            result_folder=self._tmp_folder, # concurrency issue
            result_postfix=f"{target_policy_name}_{seed}",
            silent=True
        )

        return dice


    def __call__(
        self,
        trial: optuna.trial.Trial
    ):
        reg_func_dict = {
            'p2' : DiceFunctions.P_2,
            'p3' : DiceFunctions.P_3,
            'p_3_2' : DiceFunctions.DUAL_DICE_P_3_2,
            'chi_squared' : DiceFunctions.CHI_SQUARED
        }

        dice_params = self.Parameters(
            num_layers=trial.suggest_int('num_layers', low=2, high=4),
            hidden_dim=trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            is_multihead=False,
            gamma=0.99,
            lr=trial.suggest_float('learning_rate', low=2e-07, high=1e-03, log=True),
            f1_func=DiceFunctions.DUAL_DICE_P_3_2,
            f2_func=reg_func_dict[trial.suggest_categorical('f2', list(reg_func_dict.keys()))],
            method_name='dual_dice',
            num_steps=150000,
            batch_size=8192,
            eval_iter=100
        )

        objective = []
        for name, target_actions in zip(
            self._normalized_return, 
            [self._s0_actions, self._s1_actions, self._s2_actions]
        ):
            value_history = []
            w_min = []
            w_mean = []
            w_max = []
            for seed in self._seeds:
                dice = self._fit_instance(dice_params, target_actions, name, seed)

                value_history.append(dice._experiment_data["value_per_step"])
                w = dice.predict_weights(
                    np.concatenate(self._states),
                    np.concatenate(self._actions)
                )

                w_min.append(w.min())
                w_mean.append(w.mean())
                w_max.append(w.max())

                value_artifact_id = upload_artifact(
                    artifact_store=self._artifact_store,
                    file_path=f"experiments/{self._tmp_folder}/values_{name}_{seed}.npy",
                    study_or_trial=trial
                )
                loss_plot_artifact_id = upload_artifact(
                    artifact_store=self._artifact_store,
                    file_path=f"experiments/{self._tmp_folder}/losses_{name}_{seed}.png",
                    study_or_trial=trial
                )
                value_plot_artifact_id = upload_artifact(
                    artifact_store=self._artifact_store,
                    file_path=f"experiments/{self._tmp_folder}/values_{name}_{seed}.png",
                    study_or_trial=trial
                )

                trial.set_user_attr(f"loss_plot_{name}_{seed}_aid", loss_plot_artifact_id)
                trial.set_user_attr(f"value_plot_{name}_{seed}_aid", value_plot_artifact_id)
                trial.set_user_attr(f"values_{name}_{seed}_aid", value_artifact_id)

            value_history = np.array(value_history)
        
            trial.set_user_attr(f"{name} last 100 value avg", value_history[:, -100:].mean().item())
            trial.set_user_attr(f"{name} last 200 value avg", value_history[:, -200:].mean().item())
            trial.set_user_attr(f"{name} last 300 value avg", value_history[:, -300:].mean().item())
            trial.set_user_attr(f"{name} last 300 value avg std", value_history[:, -300:].mean(axis=1).std().item())
            trial.set_user_attr(f"{name} mean w.min()", np.mean(w_min).item())
            trial.set_user_attr(f"{name} mean w.mean()", np.mean(w_mean).item())
            trial.set_user_attr(f"{name} mean w.max()", np.mean(w_max).item())

            objective.append(
                np.abs(value_history[:, -300:].mean() - self._normalized_return[name])
            )

        return np.sum(objective).item()


def main():
    args = parse_arguments()

    states = torch.load(args.states_path, weights_only=False)
    actions = torch.load(args.actions_path, weights_only=False)
    rewards = torch.load(args.rewards_path, weights_only=False)
    s0_actions = torch.load(args.s0_target_actions_path, weights_only=False)
    s1_actions = torch.load(args.s1_target_actions_path, weights_only=False)
    s2_actions = torch.load(args.s2_target_actions_path, weights_only=False)
    action_embs = torch.load(args.action_emb_path, weights_only=False)

    artifacts_folder = Path(f"./artifacts")
    artifacts_folder.mkdir(parents=True, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=str(artifacts_folder))

    study = optuna.create_study(
        sampler=RandomSampler(),
        direction="minimize",
        study_name=f"all_sasrec_dice",
        storage=JournalStorage(JournalFileBackend(file_path=f"./all_sasrec.log")),
        load_if_exists=True,
    )

    study.optimize(
        Objective(
            states=states,
            actions=actions,
            rewards=rewards,
            s0_actions=s0_actions,
            s1_actions=s1_actions,
            s2_actions=s2_actions,
            action_embs=action_embs,
            num_actions=int(args.num_actions),
            artifact_store=artifact_store,
            tmp_folder="tmp",
            device=args.device
        ),
        n_trials=10000
    )

if __name__ == "__main__":
    main()
