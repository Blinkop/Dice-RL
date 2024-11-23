import os
from tqdm import tqdm
import torch
from data import MovieLensBasicMDP, MovieLensBase


ML_PATH = './data/ml-1m.zip'
SASREC_PATH = './models/sasrec.pt'
STATES_PATH = './models/sasrec_ml_states.pt'
ACTION_DIST_PATH = './models/sasrec_actions_dist.pt'
DEVICE = torch.device('cuda:2')


@torch.no_grad()
def create_sasrec_states(
    dataset: MovieLensBase,
    sasrec_path: str,
    device: torch.device
):
    inference_sequences = {i: seq for i, seq in dataset._user_sequences.items()}

    sequences = dataset._train_df.sort_values(['userid', 'timestamp'])\
        .groupby('userid', sort=False)['itemid']\
        .apply(list)
    
    for i, seq in sequences.items():
        if i in inference_sequences:
            inference_sequences[i] = seq + inference_sequences[i]

    sasrec = torch.load(sasrec_path).to(device)
    sasrec.eval()

    states = {}
    for u, seq in tqdm(inference_sequences.items()):
        train_seq_len = len(seq) - len(dataset._user_sequences[u])

        states[u] = []
        for i in range(train_seq_len+1, len(seq)+1):
            s = torch.LongTensor(seq[:i]).to(device)
            states[u].append(sasrec.score_with_state(s)[-1].detach().cpu())

    return states


@torch.no_grad()
def create_sasrec_action_dist(
    dataset: MovieLensBase,
    sasrec_path: str,
    device: torch.device
):
    sasrec = torch.load(sasrec_path).to(device)
    sasrec.eval()

    action_dist = {}

    for u, seq in tqdm(dataset._user_sequences.items()):
        action_dist[u] = []

        for i in range(1, len(seq)+1):
            s = torch.LongTensor(seq[:i]).to(device)
            logits = sasrec.score_with_state(s)[0].flatten().detach().cpu()
            dist = torch.zeros(dataset.num_items)
            dist[logits.argmax().item()] = 1.0
            action_dist[u].append(dist)

    return action_dist


def main():
    dataset = MovieLensBasicMDP(
        num_samples=100, # doesn't matter
        policy='random', # doesn't matter
        file_path=ML_PATH
    )

    # sasrec states
    if os.path.isfile(STATES_PATH):
        raise ValueError(f'file {STATES_PATH} already exists')
    
    states = create_sasrec_states(
        dataset=dataset,
        sasrec_path=SASREC_PATH,
        device=DEVICE
    )
    torch.save(states, STATES_PATH)

    # sasrec predictions
    if os.path.isfile(ACTION_DIST_PATH):
        raise ValueError(f'file {ACTION_DIST_PATH} already exists')
    
    action_dist = create_sasrec_action_dist(
        dataset=dataset,
        sasrec_path=SASREC_PATH,
        device=DEVICE
    )
    torch.save(action_dist, ACTION_DIST_PATH)

    print('Done!')


if __name__ == '__main__':
    main()
