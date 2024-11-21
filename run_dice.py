import argparse
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from neural_dice import ValueNetwork, NeuralDice, SquaredActivation
from policy import RandomPolicy, PopularRandomPolicy
from data import AbstractDataset
from data_utils import movielens_dataset, custom_collate

import seaborn as sns
import matplotlib.pyplot as plt


NUM_THREADS = 8
NUM_WORKERS = 4

def moving_average(x, n: int = 100):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-zp", "--zeta_pos",
        help="squared zeta output",
        type=bool,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-ur", "--use_reward",
        help="count for reward in loss",
        type=bool,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument("-nr", "--norm_reg", help="lambda coefficient", type=float)
    parser.add_argument("-pr", "--primal_reg", help="alpha_Q", type=float)
    parser.add_argument("-dr", "--dual_reg", help="alpha_zeta", type=float)
    parser.add_argument("-g", "--gamma", help="discount factor", type=float)
    parser.add_argument("-hd", "--hidden_dim", help="hidden dimension size", type=int)
    parser.add_argument("-nlr", "--nu_lr", help="nu learning rate", type=float)
    parser.add_argument("-zlr", "--zeta_lr", help="zeta learning rate", type=float)
    parser.add_argument("-ds", "--dataset", help="dataset to use", type=str)
    parser.add_argument("-p", "--policy", help="policy to estimate", type=str)
    parser.add_argument("-bs", "--batch_size", help="batch size", type=int)
    parser.add_argument("-ne", "--num_episodes", help="number of episodes per batch", type=int)
    parser.add_argument("-ni", "--num_iter", help="number of iterations", type=int)
    parser.add_argument("-ei", "--eval_iter", help="evaluate every n itertion", type=int)
    parser.add_argument("-d", "--device", help="which device to use", type=str)
    parser.add_argument("-s", "--seed", help="random state seed", type=int)
    parser.add_argument("-en", "--experiment_name", help="results folder", type=str)

    return parser.parse_args()


def create_dataset(args: Namespace):
    device = torch.device(args.device)

    if args.dataset == 'movielens_basic':
        dataset = movielens_dataset(
            num_samples=args.num_episodes, sasrec_states=False, device=None
        )
    elif args.dataset == 'movielens_sasrec':
        dataset = movielens_dataset(
            num_samples=args.num_episodes, sasrec_states=True, device=device
        )
    else:
        raise ValueError(f'Unknown dataset "{args.dataset}"')

    return dataset


def create_policy(args: Namespace, dataset: AbstractDataset):
    device = torch.device(args.device)

    if args.policy == 'random':
        policy = RandomPolicy(
            num_actions=dataset.num_items,
            device=device,
            name='random',
            seed=args.seed
        )
    elif args.policy == 'randompop':
        policy = PopularRandomPolicy(
            items_count=dataset.items_count,
            device=device,
            name='pop_random',
            seed=args.seed
        )
    else:
        raise ValueError(f'Unknown policy "{args.policy}"')

    return policy


def create_dice(args: Namespace, dataset: AbstractDataset):
    device = torch.device(args.device)

    nu = ValueNetwork(
        num_layers=2,
        state_dim=dataset.state_dim,
        action_dim=dataset.num_items,
        hidden_dim=args.hidden_dim,
        output_activation=None,
        seed=args.seed
    )
    nu = nu.to(device)

    zeta = ValueNetwork(
        num_layers=2,
        state_dim=dataset.state_dim,
        action_dim=dataset.num_items,
        hidden_dim=args.hidden_dim,
        output_activation=SquaredActivation if args.zeta_pos else None,
        seed=args.seed+1
    )
    zeta = zeta.to(device)

    return NeuralDice(
        nu_network=nu,
        zeta_network=zeta,
        nu_lr=args.nu_lr,
        zeta_lr=args.zeta_lr,
        lambda_lr=args.nu_lr,
        num_actions=dataset.num_items,
        gamma=args.gamma,
        zero_reward=not args.use_reward,
        f_exponent=1.5,
        num_action_samples=None,
        primal_regularizer=args.primal_reg,
        dual_regularizer=args.dual_reg,
        norm_regularizer=args.norm_reg,
        nu_regularizer=0.0, # nu network regularizer
        zeta_regularizer=0.0, # zeta network regularizer
        weight_by_gamma=False, # weight loss by gamma**step_num
        device=device
    )


def estimate_policy(dice: NeuralDice, args: Namespace, data: AbstractDataset):
    device = torch.device(args.device)

    values = []
    for batch in data.iterate_dataset(batch_size=100):
        batch = [b.to(device) for b in batch]
        states, actions, rewards = batch

        values.append(
            dice.estimate_average_reward(
                states=states,
                actions=actions,
                rewards=rewards
            )
        )

    return sum(values) / data.num_users


def main():
    args = parse_arguments()

    torch.set_num_threads(NUM_THREADS)
    device = torch.device(args.device)

    dataset = create_dataset(args=args)
    policy = create_policy(args=args, dataset=dataset)
    dice = create_dice(args=args, dataset=dataset)
    
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        prefetch_factor=4,
        pin_memory=True,
        collate_fn=custom_collate,
        persistent_workers=False,
        generator=loader_generator
    )

    losses = []
    values = []

    for i, batch in enumerate(tqdm(loader, total=args.num_iter)):
        if i >= args.num_iter:
            break

        batch = [b.to(device) for b in batch]

        loss = dice.train_batch(batch=batch, policy=policy)
        losses.append(loss)

        if i % args.eval_iter == 0 or i == (args.num_iter - 1):
            value = estimate_policy(dice=dice, args=args, data=dataset)
            values.append(value)

    exp_folder = Path(f'experiments/{args.experiment_name}')
    exp_folder.mkdir(parents=True, exist_ok=True)

    np.save(exp_folder / 'values.npy', np.array(values))
    torch.save(dice._nu_network, exp_folder / "nu_network.pt")
    torch.save(dice._zeta_network, exp_folder / "zeta_network.pt")

    sns.set_theme()
    plt.title("loss")
    plt.plot(np.array(losses))
    plt.savefig(exp_folder / 'losses.pdf', bbox_inches="tight")
    plt.close()

    sns.set_theme()
    plt.title("values moving average")
    plt.plot(moving_average(np.array(values), n=10))
    plt.savefig(exp_folder / 'values.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
