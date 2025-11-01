import argparse

import torch

from dice import Dice, DiceFunctions
from functions import DiscreteSANetwork


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-hd",
        "--hidden_dim",
        help="hidden dimension size",
        type=int,
    )
    parser.add_argument(
        "-nl",
        "--num_layers",
        help="total number of MLP layers",
        type=int,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="learning rate",
        type=float,
    )
    parser.add_argument(
        "-f1",
        "--q_reg_func",
        help="Q regularization function",
        type=str,
    )
    parser.add_argument(
        "-f2",
        "--w_reg_func",
        help="weights regularization function",
        type=str,
    )
    parser.add_argument(
        "-mn",
        "--method_name",
        help="DICE method name",
        type=str,
    )
    parser.add_argument(
        "-ni",
        "--num_iter",
        help="number of dice iterations",
        type=int,
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
        "-d",
        "--device",
        help="device to use",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="random state seed",
        type=int,
    )
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="experiment name",
        type=str,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    states = torch.load(args.states_path, weights_only=False)
    actions = torch.load(args.actions_path, weights_only=False)
    rewards = torch.load(args.rewards_path, weights_only=False)
    target_actions = torch.load(args.target_actions_path, weights_only=False)
    action_embs = torch.load(args.action_emb_path, weights_only=False)

    state_dim = states[0].shape[1]
    action_dim = action_embs.shape[1]

    reg_func_dict = {
        'p2' : DiceFunctions.P_2,
        'p3' : DiceFunctions.P_3,
        'p_3_2' : DiceFunctions.DUAL_DICE_P_3_2,
        'chi_squared' : DiceFunctions.CHI_SQUARED,
        'kl' : DiceFunctions.KL_DIVERGENCE
    }

    q_func = DiscreteSANetwork(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        num_layers=args.num_layers,
        action_emb=action_embs,
        seed=args.seed,
        device=args.device
    )

    w_func = DiscreteSANetwork(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=action_dim,
        num_layers=args.num_layers,
        action_emb=action_embs,
        seed=args.seed,
        device=args.device
    )

    dice = Dice(
        q_function=q_func,
        w_function=w_func,
        gamma=0.99,
        q_lr=args.learning_rate,
        w_lr=args.learning_rate,
        lambda_lr=args.learning_rate,
        f1_function=reg_func_dict[args.q_reg_func],
        f2_function=reg_func_dict[args.w_reg_func],
        method_name=args.method_name,
        seed=args.seed,
        device=args.device
    )

    dice.fit(
        state=states,
        action=actions,
        reward=rewards,
        target_action=target_actions,
        num_steps=args.num_iter,
        batch_size=8192,
        eval_iter=100,
        num_workers=8,
        result_folder=args.experiment_name,
        silent=True
    )

if __name__ == "__main__":
    main()
