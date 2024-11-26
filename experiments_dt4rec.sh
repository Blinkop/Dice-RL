#!/bin/bash
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 -ds movielens_sasrec -p precalc -pt det -bs 8 -ne 1024 -ni 50000 -ei 100 -d cuda:1 -s 375 -pf models/dt4rec_actions.pt -en movielens_dualdice_dt4rec_375
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 -ds movielens_sasrec -p precalc -pt det -bs 8 -ne 1024 -ni 50000 -ei 100 -d cuda:1 -s 15378 -pf models/dt4rec_actions.pt -en movielens_dualdice_dt4rec_15378
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 -ds movielens_sasrec -p precalc -pt det -bs 8 -ne 1024 -ni 50000 -ei 100 -d cuda:1 -s 295823 -pf models/dt4rec_actions.pt -en movielens_dualdice_dt4rec_295823
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 -ds movielens_sasrec -p precalc -pt det -bs 8 -ne 1024 -ni 50000 -ei 100 -d cuda:1 -s 1121 -pf models/dt4rec_actions.pt -en movielens_dualdice_dt4rec_1121
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 -ds movielens_sasrec -p precalc -pt det -bs 8 -ne 1024 -ni 50000 -ei 100 -d cuda:1 -s 93234 -pf models/dt4rec_actions.pt -en movielens_dualdice_dt4rec_93234
