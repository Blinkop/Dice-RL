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
        "-tv",
        "--true_value",
        help="ground truth value",
        type=float,
    )
    parser.add_argument(
        "-na",
        "--num_actions",
        help="number of actions",
        type=int,
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
        "-tap",
        "--target_actions_path",
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
        "-mn",
        "--model_name",
        help="target model name",
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
        target_actions: List[np.ndarray],
        action_embs: torch.Tensor,
        true_value: float,
        num_actions: int,
        artifact_store: FileSystemArtifactStore,
        tmp_folder: str,
        device: str = "cpu"
    ):
        self._artifact_store = artifact_store
        self._tmp_folder = tmp_folder
        self._device = device
        self._seeds = [89219, 1237, 113001]

        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._target_actions = target_actions
        self._action_embs = action_embs

        self._state_dim = self._states[0].shape[1]
        self._action_dim = self._action_embs.shape[1]
        self._true_value = true_value
        self._num_actions = num_actions

    def _run_with_seed(self, params: Parameters, seed: int):
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
            target_action=self._target_actions,
            num_steps=params.num_steps,
            batch_size=params.batch_size,
            eval_iter=params.eval_iter,
            num_workers=8,
            result_folder=self._tmp_folder, # concurrency issue
            result_postfix=f'{seed}',
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
            'chi_squared' : DiceFunctions.CHI_SQUARED,
        }

        dice_params = self.Parameters(
            num_layers=trial.suggest_int('num_layers', low=2, high=4),
            hidden_dim=trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            is_multihead=False,
            gamma=0.99,
            lr=trial.suggest_float('learning_rate', low=2e-07, high=1e-03, log=True),
            f1_func=DiceFunctions.DUAL_DICE_P_3_2,
            f2_func=reg_func_dict[trial.suggest_categorical('f2', list(reg_func_dict.keys()))],
            method_name=trial.suggest_categorical('method_name', ['dual_dice', 'best_dice']),
            num_steps=500000,
            batch_size=8192,
            eval_iter=100
        )

        value_history = []
        w_min = []
        w_mean = []
        w_max = []
        for seed in self._seeds:
            dice = self._run_with_seed(dice_params, seed)

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
                file_path=f"experiments/{self._tmp_folder}/values_{seed}.npy",
                study_or_trial=trial
            )
            loss_plot_artifact_id = upload_artifact(
                artifact_store=self._artifact_store,
                file_path=f"experiments/{self._tmp_folder}/losses_{seed}.png",
                study_or_trial=trial
            )
            value_plot_artifact_id = upload_artifact(
                artifact_store=self._artifact_store,
                file_path=f"experiments/{self._tmp_folder}/values_{seed}.png",
                study_or_trial=trial
            )

            trial.set_user_attr(f"loss_plot_{seed}_artifact_id", loss_plot_artifact_id)
            trial.set_user_attr(f"value_plot_{seed}_artifact_id", value_plot_artifact_id)
            trial.set_user_attr(f"values_{seed}_artifact_id", value_artifact_id)

        value_history = np.array(value_history)
        
        trial.set_user_attr("last 100 value avg", value_history[:, -100:].mean().item())
        trial.set_user_attr("last 200 value avg", value_history[:, -200:].mean().item())
        trial.set_user_attr("last 300 value avg", value_history[:, -300:].mean().item())
        trial.set_user_attr("last 100 mean std", value_history[:, -100:].std(axis=1).mean().item())
        trial.set_user_attr("mean w.min()", np.mean(w_min).item())
        trial.set_user_attr("mean w.mean()", np.mean(w_mean).item())
        trial.set_user_attr("mean w.max()", np.mean(w_max).item())

        return np.abs(value_history[:, -300:].mean() - self._true_value).item()


def main():
    args = parse_arguments()

    states = torch.load(args.states_path, weights_only=False)
    actions = torch.load(args.actions_path, weights_only=False)
    rewards = torch.load(args.rewards_path, weights_only=False)
    target_actions = torch.load(args.target_actions_path, weights_only=False)
    action_embs = torch.load(args.action_emb_path, weights_only=False)

    artifacts_folder = Path(f"./{args.model_name}_artifacts")
    artifacts_folder.mkdir(parents=True, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=str(artifacts_folder))

    study = optuna.create_study(
        sampler=RandomSampler(),
        direction="minimize",
        study_name=f"{args.model_name}_dice_hyperparameters",
        storage=JournalStorage(JournalFileBackend(file_path=f"./{args.model_name}.log")),
        load_if_exists=True,
    )

    study.optimize(
        Objective(
            states=states,
            actions=actions,
            rewards=rewards,
            target_actions=target_actions,
            action_embs=action_embs,
            true_value=args.true_value,
            num_actions=args.num_actions,
            artifact_store=artifact_store,
            tmp_folder=f"{args.model_name}_tmp",
            device=args.device
        ),
        n_trials=10000
    )

if __name__ == "__main__":
    main()
