#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=256M
#SBATCH --array=1-50
#SBATCH --constraint="ivb|hsw"

n=$SLURM_ARRAY_TASK_ID                  # define n
args=`sed "${n}q;d" params.txt`    # get n:th line (1-indexed) of the file

module load anaconda3
source activate "\$WRKDIR/conda/envs/myenv"
srun python3 ./adult_main_anticipated.py $args
