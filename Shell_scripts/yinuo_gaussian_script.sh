#!/bin/bash 
#SBATCH --job-name=g09_1
#SBATCH --nodes 1 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb 
#SBATCH -t 300:00:00

# Going to this job's submit directory 
cd $SLURM_SUBMIT_DIR

# Defining libraries and etc
g09root="/apps"                                                                                                                                                                                                                                                                
export g09root                                                                                                                                                                                                                                                                 
. $g09root/g09/bsd/g09.profile
export GAUSS_SRCDIR=$SLURM_SUBMIT_DIR


for f in ./DA_extra_recompute/DA_0000.0001_confs*.com; do
       time g09 "$f";
done
