#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes 1
#SBATCH --output training-slurm-%j.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:gp100:1 ##added gp100 here
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

source activate ani
python training.py N
