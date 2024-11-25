#! /bin/bash
#SBATCH --job-name="run_fqe"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1-20:0:0
#SBATCH --output="logs/run_fqe2_binrew=False_svd-%a.out"
#SBATCH --constraint="type_b"
#SBATCH --array=1-10

SEED=( 230 44 480 73 575 579 191 259 336 731 )

idx=$SLURM_ARRAY_TASK_ID
echo "I am array job number" $idx

python run_fqe.py --config=config.py:SASRec2 --config.seed=${SEED[$idx-1]} --config.rank=64