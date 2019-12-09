#!/bin/bash
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=256M
#SBATCH --array=1-100
#SBATCH --constraint="ivb|hsw"

n=$SLURM_ARRAY_TASK_ID                  # define n
args=`sed "${n}q;d" params.txt`    # get n:th line (1-indexed) of the file

srun python3 mixture_main.py $args
