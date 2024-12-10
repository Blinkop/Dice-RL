#!/bin/bash
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:2 -s 27002 -sf precalc/sasrec_3_states.pt -pf precalc/sasrec_3_predictions.pt -ae precalc/sasrec_3_action_embs.pt -en movielens_dualdice_sasrec_3_27002
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:2 -s 18398 -sf precalc/sasrec_3_states.pt -pf precalc/sasrec_3_predictions.pt -ae precalc/sasrec_3_action_embs.pt -en movielens_dualdice_sasrec_3_18398
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:2 -s 35222 -sf precalc/sasrec_3_states.pt -pf precalc/sasrec_3_predictions.pt -ae precalc/sasrec_3_action_embs.pt -en movielens_dualdice_sasrec_3_35222
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:2 -s 10773 -sf precalc/sasrec_3_states.pt -pf precalc/sasrec_3_predictions.pt -ae precalc/sasrec_3_action_embs.pt -en movielens_dualdice_sasrec_3_10773
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:2 -s 26680 -sf precalc/sasrec_3_states.pt -pf precalc/sasrec_3_predictions.pt -ae precalc/sasrec_3_action_embs.pt -en movielens_dualdice_sasrec_3_26680
