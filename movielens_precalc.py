import os
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm

import torch
from torch.nn import functional as F

from src.data import MovieLens
from dt4rec_utils import make_rsa


ML_PATH = './data/ml-1m.zip'

SASREC_PRECALC_PATHS = {
    './models/sasrec.pt' : [
        './precalc/sasrec_states.pt',
        './precalc/sasrec_predictions.pt',
        './precalc/sasrec_action_embs.pt'
    ],
    './models/sasrec_2.pt' : [
        './precalc/sasrec_2_states.pt',
        './precalc/sasrec_2_predictions.pt',
        './precalc/sasrec_2_action_embs.pt'
    ],
    './models/sasrec_3.pt' : [
        './precalc/sasrec_3_states.pt',
        './precalc/sasrec_3_predictions.pt',
        './precalc/sasrec_3_action_embs.pt'
    ],
    './models/sasrec_4.pt' : [
        './precalc/sasrec_4_states.pt',
        './precalc/sasrec_4_predictions.pt',
        './precalc/sasrec_4_action_embs.pt'
    ],
}
CQL_SASREC_PRECALC_PATHS = {
    './models/cql_sasrec.pt' : [
        './precalc/cql_sasrec_states.pt',
        './precalc/cql_sasrec_predictions.pt',
        './precalc/cql_sasrec_action_embs.pt'
    ]
}
DT4REC_PRECALC_PATHS = {
    './models/dt4rec.pt' : [
        './precalc/dt4rec_states.pt',
        './precalc/dt4rec_predictions.pt',
        './precalc/dt4rec_action_embs.pt'
    ]
}
SSKNN_PRECALC_PATHS = {
    './models/ssknn.pt' : [
        './precalc/ssknn_states.pt',
        './precalc/ssknn_predictions.pt',
        './precalc/ssknn_action_embs.pt'
    ]
}

DEVICE = torch.device('cuda:1')


@torch.no_grad()
def sasrec_precalc(
    inference_sequences: Dict[int, List[int]],
    test_sequences: Dict[int, List[int]]
):
    for model_path, (
        states_path, predictions_path, action_embs_path
    ) in SASREC_PRECALC_PATHS.items():
        sasrec = torch.load(model_path).to(DEVICE)
        sasrec.eval()

        if os.path.isfile(states_path):
            print(f'file {states_path} already exists, skipping...')
        else:
            states = {}
            for u, seq in tqdm(inference_sequences.items()):
                train_seq_len = len(seq) - len(test_sequences[u])

                states[u] = []
                for i in range(train_seq_len+1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(DEVICE)
                    states[u].append(sasrec.score_with_state(s)[-1].detach().cpu())

            torch.save(states, states_path)

        if os.path.isfile(predictions_path):
            print(f'file {predictions_path} already exists, skipping...')
        else:
            predictions = {}
            for u, seq in tqdm(test_sequences.items()):
                predictions[u] = []

                for i in range(1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(DEVICE)
                    logits = sasrec.score_with_state(s)[0].flatten().detach().cpu()
                    predictions[u].append(logits[:-1].argmax().item())

            torch.save(predictions, predictions_path)
        
        if os.path.isfile(action_embs_path):
            print(f'file {action_embs_path} already exists, skipping...')
        else:
            torch.save(sasrec.item_emb.weight.data[:-1].cpu(), action_embs_path)


@torch.no_grad()
def cql_sasrec_precalc(
    inference_sequences: Dict[int, List[int]],
    test_sequences: Dict[int, List[int]]
):
    for model_path, (
        states_path, predictions_path, action_embs_path
    ) in CQL_SASREC_PRECALC_PATHS.items():
        trainer = torch.load(model_path)
        trainer.q_1 = trainer.q_1.to(DEVICE)
        trainer.q_2 = trainer.q_2.to(DEVICE)
        trainer.body = trainer.body.to(DEVICE)
        trainer.q_1.eval()
        trainer.q_2.eval()
        trainer.body.eval()

        if os.path.isfile(states_path):
            print(f'file {states_path} already exists, skipping...')
        else:
            states = {}
            for u, seq in tqdm(inference_sequences.items()):
                train_seq_len = len(seq) - len(test_sequences[u])

                states[u] = []
                for i in range(train_seq_len+1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(DEVICE)
                    states[u].append(trainer.body.score_with_state(s)[-1].detach().cpu())

            torch.save(states, states_path)

        if os.path.isfile(predictions_path):
            print(f'file {predictions_path} already exists, skipping...')
        else:
            predictions = {}
            for u, seq in tqdm(test_sequences.items()):
                predictions[u] = []

                for i in range(1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(DEVICE)
                    body_out = trainer.body.score_with_state(s)[-1]
                    body_out = body_out.reshape(-1, body_out.shape[-1])
                    out = (trainer.q_1(body_out) + trainer.q_2(body_out)) / 2.0
                    predictions[u].append(out.flatten()[:-1].argmax().item())

            torch.save(predictions, predictions_path)
        
        if os.path.isfile(action_embs_path):
            print(f'file {action_embs_path} already exists, skipping...')
        else:
            torch.save(trainer.body.item_emb.weight.data[:-1].cpu(), action_embs_path)


@torch.no_grad()
def dt4rec_precalc(
    inference_sequences: Dict[int, List[int]],
    test_sequences: Dict[int, List[int]]
):
    for model_path, (
        states_path, predictions_path, action_embs_path
    ) in DT4REC_PRECALC_PATHS.items():
        dt4rec = torch.load(model_path).to(DEVICE)
        dt4rec.eval()

        item_num = dt4rec.config.vocab_size
        seq_len = 100

        if os.path.isfile(states_path):
            print(f'file {states_path} already exists, skipping...')
        else:
            states = {}
            for u, seq in tqdm(inference_sequences.items()):
                train_seq_len = len(seq) - len(test_sequences[u])

                states[u] = []
                for i in range(train_seq_len+1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(DEVICE)
                    s = F.pad(s, (seq_len - 1 - len(s), 0), value=item_num)
                    rsa = {
                        key: value[None, ...].to(DEVICE)
                        for key, value in make_rsa(s, 3, item_num).items()
                    }
                    state = dt4rec.calc_hidden_state(**rsa)[:, -1, :].flatten()
                    states[u].append(state.detach().cpu())

            torch.save(states, states_path)

        if os.path.isfile(predictions_path):
            print(f'file {predictions_path} already exists, skipping...')
        else:
            predictions = {}
            for u, seq in tqdm(test_sequences.items()):
                predictions[u] = []

                for i in range(1, len(seq)+1):
                    s = torch.LongTensor(seq[:i]).to(DEVICE)
                    s = F.pad(s, (seq_len - 1 - len(s), 0), value=item_num)
                    rsa = {
                        key: value[None, ...].to(DEVICE)
                        for key, value in make_rsa(s, 3, item_num).items()
                    }
                    scores = dt4rec(**rsa)[:, -1, :].flatten()
                    predictions[u].append(scores[:-1].argmax().item())

            torch.save(predictions, predictions_path)
        
        if os.path.isfile(action_embs_path):
            print(f'file {action_embs_path} already exists, skipping...')
        else:
            embs = dt4rec.state_repr.item_embeddings.weight.data[:-2].cpu()
            torch.save(embs, action_embs_path)


@torch.no_grad()
def ssknn_precalc(
    inference_sequences: Dict[int, List[int]],
    test_sequences: Dict[int, List[int]],
    data_description: Dict,
    items_count: pd.Series,

):
    for model_path, (
        states_path, predictions_path, action_embs_path
    ) in SSKNN_PRECALC_PATHS.items():
        ssknn = torch.load(model_path)

        if os.path.isfile(states_path):
            print(f'file {states_path} already exists, skipping...')
        else:
            states = {}
            for u, seq in tqdm(inference_sequences.items()):
                train_seq_len = len(seq) - len(test_sequences[u])

                states[u] = []
                for i in range(train_seq_len+1, len(seq)+1):
                    s = seq[:i]
                    d = pd.DataFrame({
                        data_description['items'] : s,
                        data_description['order'] : np.arange(len(s))
                    })
                    d[data_description['users']] = u
                    state = ssknn.get_current_state(
                        user_test_interactions=d,
                        item_popularity=items_count,
                        data_description=data_description,
                        calculate_subseqs=False,
                        top_pop=300
                    )
                    states[u].append(torch.FloatTensor(state.ravel()))

            torch.save(states, states_path)

        if os.path.isfile(predictions_path):
            print(f'file {predictions_path} already exists, skipping...')
        else:
            predictions = {}
            for u, seq in tqdm(test_sequences.items()):
                predictions[u] = []

                for i in range(1, len(seq)+1):
                    s = seq[:i]
                    d = pd.DataFrame({'itemid' : s, 'timestamp' : np.arange(len(s))})
                    d['userid'] = u
                    scores = ssknn.recommend(d, data_description).ravel()
                    predictions[u].append(scores.argmax())

            torch.save(predictions, predictions_path)
        
        if os.path.isfile(action_embs_path):
            print(f'file {action_embs_path} already exists, skipping...')
        else:
            action_embs = torch.FloatTensor(ssknn.get_action_embeddings(data_description))
            torch.save(action_embs, action_embs_path)


def main():
    train_df, test_df, holdout_df, description = MovieLens.create_dataset(ML_PATH)
    test_sequences = MovieLens.create_sequences(test_df, holdout_df)

    items_count = train_df['itemid'].value_counts()

    inference_sequences = {i: seq for i, seq in test_sequences.items()}

    train_sequences = train_df.sort_values(['userid', 'timestamp'])\
        .groupby('userid', sort=False)['itemid']\
        .apply(list)
    
    for i, seq in train_sequences.items():
        if i in inference_sequences:
            inference_sequences[i] = seq + inference_sequences[i]

    sasrec_precalc(inference_sequences, test_sequences)
    cql_sasrec_precalc(inference_sequences, test_sequences)
    dt4rec_precalc(inference_sequences, test_sequences)
    ssknn_precalc(inference_sequences, test_sequences, description, items_count)

    print('Done!')


if __name__ == '__main__':
    main()
