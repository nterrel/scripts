#!/bin/bash
#SBATCH --job-name=p1cpu
#SBATCH --nodes 1
#SBATCH --output pickle-slurm-time-%j.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:gp100:1 ##added gp100 here
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

source activate ani
python pickle-dataset-assign-res-time.py
