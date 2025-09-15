import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

import torch
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mn",
        "--model_name",
        help="model alias to use for file naming",
        type=str,
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        help="log file to preprocess",
        type=str,
    )
    parser.add_argument(
        "-sp",
        "--sasrec_path",
        help="sasrec path",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="results folder",
        type=str
    )
    parser.add_argument(
        "-d",
        "--device",
        help="device to use",
        type=str,
    )

    return parser.parse_args()

@torch.no_grad()
def sasrec_get_state(
    seq: List[int],
    model: torch.nn.Module,
    device: str
):
    s = torch.LongTensor(seq).to(device)
    return model.score_with_state(s)[-1].detach().cpu().numpy()

@torch.no_grad()
def sasrec_prediction(
    seq: List[int],
    model: torch.nn.Module,
    device: str
):
    seq_tensor = torch.LongTensor(seq).to(device)
    logits = model.score(seq_tensor).flatten().detach().cpu()[:-1]
    logits[seq] = logits.min()

    return logits.argmax().item()


def main():
    args = parse_arguments()

    results_folder = Path(args.folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path).sort_values(by='timestamp').reset_index()

    sasrec = torch.load(args.sasrec_path, weights_only=False).to(args.device)
    sasrec.eval()

    action_embs = sasrec.item_emb.weight.data[:-1].cpu()

    sequences = df.groupby('userid')['movieid'].apply(np.array)
    timestamps = df.groupby('userid')['timestamp'].apply(np.array)
    sources = df.groupby('userid')['source'].apply(np.array)
    ratings = df.groupby('userid')['rating'].apply(np.array)

    states = []
    actions = []
    rewards = []
    target_actions = []

    for _, (u, seq) in tqdm(enumerate(sequences.items()), total=len(sequences)):
        policy_mask = sources[u] == 'policy'
        positive_mask = ratings[u] > 0

        policy_seq = seq[policy_mask]
        policy_timestamps = timestamps[u][policy_mask]
        policy_ratings = ratings[u][policy_mask]

        positive_seq = seq[positive_mask]
        positive_timestamps = timestamps[u][positive_mask]

        states.append(np.zeros((len(policy_seq), action_embs.shape[1])))
        actions.append(np.zeros(len(policy_seq)))
        rewards.append(np.zeros(len(policy_seq)))
        target_actions.append(np.zeros(len(policy_seq)))

        for j in range(len(policy_seq)):
            t = policy_timestamps[j]
            state_seq = positive_seq[positive_timestamps < t]

            states[-1][j] = sasrec_get_state(
                seq=state_seq, model=sasrec, device=args.device
            )
            actions[-1][j] = policy_seq[j]
            rewards[-1][j] = policy_ratings[j]
            target_actions[-1][j] = sasrec_prediction(
                seq=state_seq, model=sasrec, device=args.device
            )

    torch.save(actions,        f'{args.folder}/actions.pt')
    torch.save(rewards,        f'{args.folder}/rewards.pt')
    torch.save(states,         f'{args.folder}/{args.model_name}_states.pt')
    torch.save(target_actions, f'{args.folder}/{args.model_name}_actions.pt')
    torch.save(action_embs,    f'{args.folder}/{args.model_name}_action_embs.pt')

if __name__ == "__main__":
    main()
