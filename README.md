# Dice-RL

To pre-calculate sasrec states and predictions run:

```
python movielens_precalc.py
```

To run experiment use ```run_dice.py``` script. For example:

```
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 -ds movielens_sasrec -p precalc -pt det -bs 8 -ne 1024 -ni 50000 -ei 100 -d cuda:2 -s 93234 -pf models/sasrec_2_actions.pt -en movielens_dualdice_sasrec_2_93234
```

Parameters information:

```
python run_dice.py --help
```
