#!/bin/bash
#SBATCH --job-name=1x_forces
#SBATCH --nodes 1
#SBATCH --output 1x_forces.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:gp100:1 ##added gp100 here
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

conda activate cuaev
python test_1x_forces.py
