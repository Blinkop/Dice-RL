import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from cartpole import RandomPolicy
from data import CartPoleMDP, MovieLensBasicMDP
from neural_dice_v2 import NeuralDice, SquaredActivation, ValueNetwork
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

RANDOM_SEED = 12
DEVICE = torch.device("cpu")
BATCH_SIZE = 2
NUM_SAMPLES = 1024
NUM_ITER = 10000
EVAL_ITER = 100

ZETA_POS = False  # squared output of zeta(s,a) (used in GenDICE, BestDice)
ZERO_REWARD = True  # whether include the reward in loss or not (used in MQL, BestDICE)
NORM_REGULARIZER = 0.0  # use lambda or not (used in GenDICE, GradientDICE, BestDICE)
PRIMAL_REGULARIZER = 0.0  # alpha_Q (used in GenDICE, GradientDICE)
DUAL_REGULARIZER = 1.0  # alpha_zeta (used in DualDICE, BestDICE)

NU_LR = 0.0001
ZETA_LR = 0.0001
GAMMA = 0.99


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def custom_collate(data_list):
    first_state = []
    current_state = []
    current_action = []
    next_state = []
    rewards = []
    step_num = []

    for fs, cs, ca, ns, rw, sn in data_list:
        first_state.append(fs)
        current_state.append(cs)
        current_action.append(ca)
        next_state.append(ns)
        rewards.append(rw)
        step_num.append(sn)

    return (
        torch.concat(first_state, dim=0),
        torch.concat(current_state, dim=0),
        torch.concat(current_action, dim=0),
        torch.concat(next_state, dim=0),
        torch.concat(rewards, dim=0),
        torch.concat(step_num, dim=0),
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def estimate_policy(dice: NeuralDice, data: MovieLensBasicMDP, print_zeta=False):
    values = []
    for batch in data.iterate_dataset(batch_size=100):
        batch = [b.to(DEVICE) for b in batch]
        states, actions, rewards = batch

        values.append(
            dice.estimate_average_reward(
                states=states, actions=actions, rewards=rewards, print_zeta=print_zeta
            )
        )
        print_zeta = False

    return np.mean(values)


def calc_grad(model):
    return np.sqrt(sum([torch.norm(p.grad) ** 2 for p in model.parameters()]))


def main():
    argv = sys.argv
    exp_name = argv[1]

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    dataset = CartPoleMDP(
        ds_path="/home/hdilab/amgimranov/Dice-RL/cartpole_ds.npy",
        num_samples=NUM_SAMPLES,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        collate_fn=custom_collate,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    policy = RandomPolicy(
        device=DEVICE,
        name="RandomPolicy",
    )

    nu = ValueNetwork(
        num_layers=2,
        state_dim=4,
        action_dim=2,
        hidden_dim=64,
        output_activation=None,
        seed=RANDOM_SEED,
    )
    nu = nu.to(DEVICE)

    zeta = ValueNetwork(
        num_layers=2,
        state_dim=4,
        action_dim=2,
        hidden_dim=64,
        output_activation=SquaredActivation if ZETA_POS else None,
        seed=RANDOM_SEED + 1,
    )
    zeta = zeta.to(DEVICE)

    dice = NeuralDice(
        nu_network=nu,
        zeta_network=zeta,
        nu_lr=NU_LR,
        zeta_lr=ZETA_LR,
        lambda_lr=NU_LR,
        num_actions=2,
        gamma=GAMMA,
        zero_reward=ZERO_REWARD,
        f_exponent=2,
        primal_form=False,
        num_action_samples=None,  # number of action samples used in Q expectation
        primal_regularizer=PRIMAL_REGULARIZER,
        dual_regularizer=DUAL_REGULARIZER,
        norm_regularizer=NORM_REGULARIZER,
        nu_regularizer=0.0,  # nu network regularizer
        zeta_regularizer=0.0,  # zeta network regularizer
        weight_by_gamma=False,  # weight loss by gamma**step_num
        device=DEVICE,
    )

    losses = []
    values = []

    for i, batch in enumerate(tqdm(loader, total=NUM_ITER)):
        if i >= NUM_ITER:
            break

        batch = [b.to(DEVICE) for b in batch]

        loss = dice.train_batch(batch=batch, policy=policy)
        losses.append(loss)

        if i % EVAL_ITER == 0 or i == (NUM_ITER - 1):
            value = estimate_policy(dice=dice, data=dataset, print_zeta=True)
            values.append(value)
            print(f"Value on iteration {i}: {value}. Loss: {loss}")
            print("nu grad:", calc_grad(nu))
            print("zeta grad:", calc_grad(zeta))

    exp_folder = Path(f"experiments/{exp_name}")
    exp_folder.mkdir(parents=True, exist_ok=False)

    np.save(exp_folder / "values.npy", np.array(values))
    np.save(exp_folder / "losses.npy", np.array(losses))
    torch.save(zeta, exp_folder / "zeta.pickle")
    torch.save(nu, exp_folder / "nu.pickle")

    sns.set_theme()
    plt.title("values moving_average(100)")
    plt.plot(moving_average(np.array(values), n=10))
    plt.savefig(exp_folder / "values.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.title("losses")
    plt.plot(np.array(losses))
    plt.savefig(exp_folder / "losses.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
