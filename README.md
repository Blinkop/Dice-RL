# Dice-RL

## Repository structure

* ```data/```: raw data folder
* ```models/```: stores trained target models
* ```src/```: DICE source code
* ```experiments_*.sh```: MovieLens experiments scripts
* ```*.ipynb```: visualize MovieLens results
* ```run_dice.py```: DICE script
* ```dataset_precalc.py```: Pre-calculation script
* ```model.py, ssknn.py, gpt.py, dt4rec_utils.py, cql_dqn.py, RECE/```: Target models` source code used in MovieLens pre-calculation script

## DICE script parameters

Parameters information:

```
python run_dice.py --help
```

## Useful information

This DICE implementation uses pre-calculated states, actions and action embeddings in order to speed up the estimation process. Therefore as an input, the ```run_dice.py``` script takes files
with pre-calculated states, actions and embeddings and doesn`t use the target model directly.

## How to run the MovieLens experiments

To pre-calculate states, predictions, and action embeddings and save them into ```precalc``` directory run:

```
python dataset_precalc.py
```

To reproduce experiments for MovieLens with models' own states and multihead $\nu$ and $\zeta$ architecture run ```experiments_*.sh```. It may require to change the device to run on (see the script).

The final plot is drawn with ```movielens_result.ipynb``` notebook.

## Running DICE with your own dataset

NOTE, that this repository comes with already trained models for MovieLens dataset under ```models``` directory, and a ready to use pre-calculation script ```dataset_precalc.py```, which will
compute states, actions and action embeddings for all of the presented models. In order to use your own dataset your must train models and compute states, actions and action embedding on your
side. Take ```dataset_precalc.py``` script as an useful example for implementing your own pre-calculation logic.

When states, actions and action embeddings are ready, the respected files are passed into ```run_dice.py``` script. One thing left is to pass the dataset path into ```run_dice.py``` script
using ```-ds [PATH]``` key. Dataset should be in ```.csv``` format and at least have columns ```userid, itemid, timestamp``` (```timestamp``` should contain integer values). If the dataset is
large, it is recommended to change ```test_size, validation_size, q``` parameters in the ```src/data/data.py/DICEDataset.create_dataset/get_dataset()``` call.

To run DICE experiment use ```run_dice.py``` script. The results will be written to ```experiments``` folder. For example:

```
python run_dice.py --no-zeta_pos --no-use_reward -nr 0.0 -pr 0.0 -dr 1.0 -g 0.99 -hd 64 -nlr 0.00001 -zlr 0.00001 --lr_schedule --multihead -ds ./data/ml-1m.zip -bs 8 -ne 1024 -ni 200000 -ei 100 -d cuda:2 -s 27002 -sf precalc/sasrec_2_states.pt -pf precalc/sasrec_2_predictions.pt -ae precalc/sasrec_2_action_embs.pt -en movielens_dualdice_sasrec_2_27002
```
