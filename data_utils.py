import torch

from polara import get_movielens_data

from data import MovieLensBasicMDP, MovieLensSasrecMDP


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map

def custom_collate(data_list):
    first_state = []
    current_state = []
    current_action = []
    next_state = []
    rewards = []
    step_num = []
    has_next = []

    for fs, cs, ca, ns, rw, sn, hn in data_list:
        first_state.append(fs)
        current_state.append(cs)
        current_action.append(ca)
        next_state.append(ns)
        rewards.append(rw)
        step_num.append(sn)
        has_next.append(hn)

    return (
        torch.concat(first_state, dim=0),
        torch.concat(current_state, dim=0),
        torch.concat(current_action, dim=0),
        torch.concat(next_state, dim=0),
        torch.concat(rewards, dim=0),
        torch.concat(step_num, dim=0),
        torch.concat(has_next, dim=0)
    )


def movielens_dataset(
    num_samples: int,
    sasrec_states: bool = False,
    device: torch.device = torch.device('cpu'),
):
    data = get_movielens_data(local_file='./data/ml-1m.zip', include_time=True)
    data, _ = transform_indices(data, 'userid', 'movieid')

    sequences = data.sort_values(['userid', 'timestamp'])\
        .groupby('userid', sort=False)['movieid']\
        .apply(list)

    if sasrec_states:
        dataset = MovieLensSasrecMDP(
            num_items=len(data['movieid'].unique()),
            num_samples=num_samples,
            user_sequences=sequences.values.tolist(),
            sasrec_path=None,
            sasrec_device=device,
            embeddings_path='./models/sasrec_ml_states.pt'
        )
    else:
        dataset = MovieLensBasicMDP(
            num_items=len(data['movieid'].unique()),
            num_samples=num_samples,
            user_sequences=sequences.values.tolist()
        )
    
    return dataset
