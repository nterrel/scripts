#!/bin/bash
#SBATCH --job-name=1x_no_bias
#SBATCH --nodes 1
#SBATCH --output train.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:gp100:1 ##added gp100 here
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

conda activate cuaev
python ex_force_training.py
