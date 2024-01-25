#!/bin/bash
#SBATCH --job-name=2x_first
#SBATCH --nodes 1
#SBATCH --output 2x-n1c_compute.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:gp100:1 ##added gp100 here
#SBATCH --mem=40G
#SBATCH -t 20:00:00

cd $SLURM_SUBMIT_DIR
source miniconda3/etc/profile.d/conda.sh
conda activate cuaev
python WIP_n1c.py
