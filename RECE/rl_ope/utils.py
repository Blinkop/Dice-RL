from RECE.data import get_dataset, data_to_sequences, data_to_sequences_rating
from RECE.eval_utils import sasrec_model_scoring

from time import time
from functools import reduce
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds, LinearOperator

import numpy as np
import torch

from RECE.data import get_dataset, data_to_sequences, SequentialDataset
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def process_subseqs(subseqs_h, model_D_sasrec, device):
    states = []
    next_states = []
    scores = []
    actions = []
    ratings = []
    for at, subseq, rating in subseqs_h:
        with torch.no_grad():
            score, state = model_D_sasrec.score_with_state(torch.tensor(subseq, device=device, dtype=torch.long))
            states.append(state.detach().cpu().numpy())

            next_subseq = subseq[1:]
            next_subseq.append(at)
            _, next_state = model_D_sasrec.score_with_state(torch.tensor(next_subseq, device=device, dtype=torch.long))
            next_states.append(next_state.detach().cpu().numpy())
            # scores.append(score.detach().cpu().numpy())
            actions.append(at)
            ratings.append(rating)

    return states, next_states, actions, ratings, scores

def prepare_svd(data, data_description, rank, device):
    userid = data_description['users']
    itemid = data_description['items']

    n_users = len(data[userid].unique())
    n_items = data_description['n_items'] + 2 # +1 for the pad item of SasRec and +1 for &@!&%$

    interaction_matrix = csr_matrix(
                                    (
                                        data['rating'],
                                        (data[userid], data[itemid]), #subtract 1 to start with [0,1,2,...]
                                    ),
                                    shape=(n_users, n_items),
                                    dtype=float)

    _, singular_values, vh = svds(
        interaction_matrix,
        k=rank,
        return_singular_vectors='vh'
    )

    sort_order = np.argsort(singular_values)[::-1]
    # import matplotlib.pyplot as plt
    # plt.plot(singular_values[::-1])
    # plt.semilogy(singular_values[::-1], 'k-')
    item_factors = torch.from_numpy(np.ascontiguousarray(vh[sort_order, :].T, dtype=np.float32)).to(device)

    return item_factors

def extract_states_actions(data, model_D_sasrec, num_embeddings, n, data_description, pad_token, device, n_neg_samples=None):
    # model_D_sasrec = model_D_sasrec.to(device)
    states = []
    next_states = []
    actions = []
    scores = []
    ratings = []

    full_sequences = data_to_sequences_rating(data, data_description, n_neg_samples)

    seqs = []
    for _, h in tqdm(full_sequences.items(), total=len(full_sequences)):
        seqt = h.copy()
        for i in range(len(h)-1):
            (at, rating) = seqt.pop()
            actions.append(at)
            ratings.append(rating)
            seqs.append(list(list(zip(*seqt[-n:]))[0]))

    seqs_padded = [np.pad(arr, (n - len(arr), 0), mode='constant', constant_values=pad_token) for arr in seqs]
    seqs_padded = torch.tensor(seqs_padded, device=device)[:, -num_embeddings:]
    with torch.no_grad():
        states = model_D_sasrec.log2feats(seqs_padded)[:, -1].cpu().detach().numpy()

    next_seqs_padded = [np.pad(arr[1:] + [actions[i]],
                        (n - len(arr), 0), mode='constant',
                        constant_values=pad_token) for i, arr in enumerate(seqs)]
    next_seqs_padded = torch.tensor(next_seqs_padded, device=device)[:, -num_embeddings:]
    with torch.no_grad():
        next_states = model_D_sasrec.log2feats(next_seqs_padded)[:, -1].cpu().detach().numpy()

    return states, next_states, actions, ratings, scores, full_sequences


def process_seq(seqt, model_D_sasrec, device):
    action, seq, rating = seqt

    with torch.no_grad():
        score, state = model_D_sasrec.score_with_state(torch.tensor(seq, device=device, dtype=torch.long))
        state = state.detach().cpu().numpy()

    return state, action, rating, score

def extract_states_actions_val(data, model_D_sasrec, item_embs, n, data_description, pad_token, device):
    # model_D_sasrec = model_D_sasrec.to(device)
    states = []
    actions = []
    scores = []
    ratings = []

    full_sequences = data_to_sequences_rating(data, data_description)

    for _, seqt in tqdm(full_sequences.items(), total=len(full_sequences)):
        (at, rating) = seqt[-1]
        if len(seqt) < n + 1:
            seqt = np.pad(seqt, (n - len(seqt), 0), mode='constant', constant_values=pad_token)

        seqt = (at, list(zip(*seqt[-n-1:-1]))[0], rating)

        state, action, rating, score = process_seq(seqt, model_D_sasrec, device)
        states.append(state)
        actions.append(action)
        ratings.append(rating)
        scores.append(score)

    return states, actions, ratings, scores