#!/bin/bash
#SBATCH --job-name=ch4_optimization
#SBATCH --nodes 1
#SBATCH --output opt_ch4.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH -t 60:00:00

cd $SLURM_SUBMIT_DIR

conda activate cuaev
python opt_ch4.py
