#!/bin/bash
#SBATCH --job-name=test_set_optimization
#SBATCH --nodes 1
#SBATCH --output opt.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:gp100:1
#SBATCH --mem=50G
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

conda activate cuaev
python opt.py
