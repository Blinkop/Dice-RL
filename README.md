# Dice-RL

First, put dataset into ```data``` folder. Dataset should be in ```.csv``` format and at least have columns ```userid, itemid, timestamp``` (```timestamp``` should contain integer values).

If the dataset is large, it is recommended to change ```test_size, validation_size, q``` parameters in the ```src/data/data.py/MovieLens.create_dataset()/get_dataset()``` function.

To pre-calculate states, predictions, and action embeddings for selected dataset (MovieLens-1M by default), change data and saved model paths in ```dataset_precalc.py``` and run:

```
python dataset_precalc.py
```

To run experiment for DICE use ```run_dice.py``` script. The results will be written to ```experiments``` folder. For example:

```
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:0 -s 27002 -sf precalc/cql_sasrec_states.pt -pf precalc/cql_sasrec_predictions.pt -ae precalc/cql_sasrec_action_embs.pt -en movielens_dualdice_cql_sasrec_27002
```

Parameters information:

```
python run_dice.py --help
```

To reproduce experiments for MovieLens with models' own states and multihead $\nu$ and $\zeta$ architecture run ```experiments_*.sh```. It may require to change the device to run on.

The final plot is drawn with ```movielens_result.ipynb``` notebook.

