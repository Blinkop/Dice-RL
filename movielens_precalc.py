import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from src.data import MovieLensBasicMDP, MovieLensBase
from dt4rec_utils import make_rsa


ML_PATH = './data/ml-1m.zip'
SASRECS_AND_PREDICTIONS = [
    ('./models/sasrec_2.pt', './models/sasrec_2_actions.pt'),
    ('./models/sasrec_3.pt', './models/sasrec_3_actions.pt'),
    ('./models/sasrec_4.pt', './models/sasrec_4_actions.pt'),
]
CQL_AND_PREDICTIONS = [
    ('./models/cql_sasrec.pt', './models/cql_sasrec_actions.pt')
]
DT4REC_AND_PREDICTIONS = [
    ('./models/dt4rec.pt', './models/dt4rec_actions.pt')
]
SSKNN_AND_PREDICTIONS = [
    ('./models/ssknn.pt', './models/ssknn_actions.pt')
]
SASREC_STATES_PATH = './models/sasrec.pt'
STATES_PATH = './models/sasrec_ml_states.pt'
DEVICE = torch.device('cuda:1')


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
def create_sasrec_predictions(
    dataset: MovieLensBase,
    sasrec_path: str,
    device: torch.device
):
    sasrec = torch.load(sasrec_path).to(device)
    sasrec.eval()

    predictions = {}

    for u, seq in tqdm(dataset._user_sequences.items()):
        predictions[u] = []

        for i in range(1, len(seq)+1):
            s = torch.LongTensor(seq[:i]).to(device)
            logits = sasrec.score_with_state(s)[0].flatten().detach().cpu()
            predictions[u].append(logits.argmax().item())

    return predictions

@torch.no_grad()
def create_cqlsasrec_predictions(
    dataset: MovieLensBase,
    cql_path: str,
    device: torch.device
):
    trainer = torch.load(cql_path)
    trainer.q_1 = trainer.q_1.to(device)
    trainer.q_2 = trainer.q_2.to(device)
    trainer.body = trainer.body.to(device)
    trainer.q_1.eval()
    trainer.q_2.eval()
    trainer.body.eval()

    predictions = {}

    for u, seq in tqdm(dataset._user_sequences.items()):
        predictions[u] = []

        for i in range(1, len(seq)+1):
            s = torch.LongTensor(seq[:i]).to(device)
            body_out = trainer.body.score_with_state(s)[-1]
            body_out = body_out.reshape(-1, body_out.shape[-1])
            out = (trainer.q_1(body_out) + trainer.q_2(body_out)) / 2.0
            predictions[u].append(out.argmax().item())

    return predictions


@torch.no_grad()
def create_dt4rec_predictions(
    dataset: MovieLensBase,
    dt4rec_path: str,
    device: torch.device
):
    dt4rec = torch.load(dt4rec_path).to(device)
    dt4rec.eval()

    item_num = dt4rec.config.vocab_size
    seq_len = 100

    predictions = {}

    for u, seq in tqdm(dataset._user_sequences.items()):
        predictions[u] = []

        for i in range(1, len(seq)+1):
            s = torch.LongTensor(seq[:i]).to(device)
            s = F.pad(s, (seq_len - 1 - len(s), 0), value=item_num)
            rsa = {
                key: value[None, ...].to(device)
                for key, value in make_rsa(s, 3, item_num).items()
            }
            state = dt4rec(**rsa)
            # [:-1] to fix a bug
            predictions[u].append(state[:, -1, :].flatten()[:-1].argmax().item())

    return predictions


@torch.no_grad()
def create_ssknn_predictions(
    dataset: MovieLensBase,
    ssknn_path: str
):
    ssknn = torch.load(ssknn_path)

    data_description = {
        'users': 'userid',
        'items': 'itemid',
        'order': 'timestamp',
        'n_users': 5400,
        'n_items': dataset.num_items
    }

    predictions = {}

    for u, seq in tqdm(dataset._user_sequences.items()):
        predictions[u] = []

        for i in range(1, len(seq)+1):
            s = seq[:i]
            d = pd.DataFrame({'itemid' : s, 'timestamp' : np.arange(len(s))})
            d['userid'] = u
            scores = ssknn.recommend(d, data_description).ravel()
            predictions[u].append(scores.argmax())

    return predictions


def main():
    dataset = MovieLensBasicMDP(
        num_samples=100, # doesn't matter
        policy='random', # doesn't matter
        file_path=ML_PATH
    )

    # sasrec states
    if os.path.isfile(STATES_PATH):
        print(f'file {STATES_PATH} already exists, skipping...')
    else:
        states = create_sasrec_states(
            dataset=dataset,
            sasrec_path=SASREC_STATES_PATH,
            device=DEVICE
        )
        torch.save(states, STATES_PATH)

    # sasrec predtctions
    for model_path, pred_path in SASRECS_AND_PREDICTIONS:
        if os.path.isfile(pred_path):
            print(f'file {pred_path} already exists, skipping {model_path} model ...')
            continue
    
        predictions = create_sasrec_predictions(
            dataset=dataset,
            sasrec_path=model_path,
            device=DEVICE
        )
        torch.save(predictions, pred_path)

    # dt4rec predtctions
    for model_path, pred_path in DT4REC_AND_PREDICTIONS:
        if os.path.isfile(pred_path):
            print(f'file {pred_path} already exists, skipping {model_path} model ...')
            continue
    
        predictions = create_dt4rec_predictions(
            dataset=dataset,
            dt4rec_path=model_path,
            device=DEVICE
        )
        torch.save(predictions, pred_path)

    # CQL SASRec prediction
    for model_path, pred_path in CQL_AND_PREDICTIONS:
        if os.path.isfile(pred_path):
            print(f'file {pred_path} already exists, skipping {model_path} model ...')
            continue
    
        predictions = create_cqlsasrec_predictions(
            dataset=dataset,
            cql_path=model_path,
            device=DEVICE
        )
        torch.save(predictions, pred_path)

    # SSKNN prediction
    for model_path, pred_path in SSKNN_AND_PREDICTIONS:
        if os.path.isfile(pred_path):
            print(f'file {pred_path} already exists, skipping {model_path} model ...')
            continue
    
        predictions = create_ssknn_predictions(
            dataset=dataset,
            ssknn_path=model_path
        )
        torch.save(predictions, pred_path)

    print('Done!')


if __name__ == '__main__':
    main()
