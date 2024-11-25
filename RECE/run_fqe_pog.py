from absl import app
from absl import flags
from ml_collections import config_flags

from data import get_dataset, data_to_sequences, data_to_sequences_rating
from eval_utils import sasrec_model_scoring

from time import time
from functools import reduce
import uuid
from tqdm import tqdm
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import SASRec
from data import get_dataset, data_to_sequences, SequentialDataset
from utils import topn_recommendations, downvote_seen_items
from eval_utils import model_evaluate, sasrec_model_scoring, get_test_scores

from sklearn.linear_model import LogisticRegression
from train import build_sasrec_model, prepare_sasrec_model
from concurrent.futures import ProcessPoolExecutor
from rl_ope.fqe import RLDatasetOnline, RLDatasetOnlineVal, FQE
from rl_ope.utils import prepare_svd, extract_states_actions, extract_states_actions_val
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def main(_):
    args = FLAGS.config

    device = args.device
    seed = args.seed

    values_path = f"./saved_values/values{args.alg_type}_pen={args.optim_conf.weight_decay},"\
                  f"bs={args.fqe_params.bs},"\
                  f"nneg={args.n_neg_samples},"\
                  f"subseq_len={args.subseq_len},"\
                  f"binary_rew={args.binary_rew},"\
                  f"seed={seed}"\
                  f"rank={args.rank}.npy"

    fqe_w_path = f"./saved_fqes/fqe{args.alg_type}_pen={args.optim_conf.weight_decay},"\
                 f"bs={args.fqe_params.bs},"\
                 f"nneg={args.n_neg_samples},"\
                 f"subseq_len={args.subseq_len},"\
                 f"binary_rew={args.binary_rew},"\
                 f"seed={seed}"\
                 f"rank={args.rank}.pt"

    data_path = "/home/svsamsonov/hdiRecSysIL/RECE/mv1m/ml-1m.zip"


    training_temp, data_description_temp, testset_valid_temp, _, holdout_valid_temp, _ = get_dataset(local_file=data_path,
                                                                                         splitting='temporal_full',
                                                                                         q=0.8)

    training_full, data_description_full, testset_valid_full, _, holdout_valid_full, _ = get_dataset(local_file=data_path,
                                                                                         splitting='full',
                                                                                         q=0.8)

    s = training_full['rating']
    if args.binary_rew:
        training_full['rating'] = s.where(s >= 3, 0).mask(s >= 3, 1)
    else:
        training_full['rating'] = np.ones(s.shape[0], dtype=np.int32)

    #Load model_D_sasrec and model_e
    model_e, _, _, _ = globals()[args.config_e.gen_model](args.config_e.params, training_temp, data_description_temp, device)
    model_e.load_state_dict(torch.load(args.config_e.chkpt_path, map_location=torch.device(device)))

    n_actions = model_e.item_num + 1
    all_actions = np.arange(n_actions, dtype=np.int32)

    model_D, _, _, _ = globals()[args.config_D.gen_model](args.config_D.params, training_full, data_description_full, device)
    model_D.load_state_dict(torch.load(args.config_D.chkpt_path, map_location=torch.device(device)))

    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    item_embs = None
    if args.rank > 0:
        item_embs = prepare_svd(training_full, data_description_full, args.rank, device)

    states, next_states, actions, ratings, _, full_sequences = extract_states_actions(training_full,
                                                                                      model_e,
                                                                                      item_embs,
                                                                                      args.subseq_len,
                                                                                      data_description_full,
                                                                                      "cpu",
                                                                                      args.n_neg_samples)

    states_val, actions_val, ratings_val, _ = extract_states_actions_val(testset_valid_full,
                                                                         model_e,
                                                                         item_embs,
                                                                         args.subseq_len_val,
                                                                         data_description_full,
                                                                         "cpu")

    full_sequences_val = data_to_sequences(testset_valid_full, data_description_full)

    state_dim = len(states[0])

    dataset_config = {
        "states": states,
        "next_states": next_states,
        "actions": actions,
        "rewards": ratings,
        "all_actions": all_actions,
        "full_sequences": full_sequences,
        "n": args.subseq_len,
        "pad_token": None,
        "n_neg_samples": args.n_neg_samples,
    }

    dataset_val_config = {
        "states": states_val,
        "actions": actions_val,
        "rewards": ratings_val,
        "full_sequences": full_sequences_val,
        "n": args.subseq_len_val,
        "pad_token": None
    }

    dataset_config["pad_token"] = model_e.pad_token
    dataset_val_config["pad_token"] = model_e.pad_token

    dataset = RLDatasetOnline(dataset_config)
    val_dataset = RLDatasetOnlineVal(dataset_val_config)

    fqe = FQE(dataset,
              val_dataset,
              model_e,
              args.optim_conf,
              args.fqe_params.n_epochs,
              state_dim,
              n_actions,
              args.fqe_params.hidden_size,
              gamma=args.gamma,
              device=device)

    values = fqe.train(batch_size=args.fqe_params.bs, plot_info=False)

    torch.save(fqe.q.state_dict(), fqe_w_path)

    np.save(values_path, values)


if __name__ == "__main__":
    # python run_fqe.py --config=config.py:SASRec --config.config_e.chkpt_path=./saved_models/model_e0.pt --config.values_path=./saved_values/values_0.npy
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config")

    app.run(main)















