#! /bin/bash
#SBATCH --job-name="run_fqe"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1-20:0:0
#SBATCH --output="logs/eval.out"
#SBATCH --constraint="type_b"

# python run_fqe_pog.py --config=config.py:SASRec2 --config.optim_conf.weight_decay=0.00001
python evaluation.py
