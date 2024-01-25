#!/bin/bash
#SBATCH --job-name=cos_sim_1x
#SBATCH --nodes 1
#SBATCH --output ani1x-cos_sim-magnitudes.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:gp100:1 ##added gp100 here
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

conda activate cuaev
python ani1x-cos_sim-magnitudes.py
