#!/bin/bash 
#SBATCH --job-name=gau1 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb 
#SBATCH -t 20:00:00

# Going to this job's submit directory 
cd $SLURM_SUBMIT_DIR

# Defining libraries and etc
g09root="/apps"                                                                                                                                                                                                                                                                
export g09root                                                                                                                                                                                                                                                                 
. $g09root/g09/bsd/g09.profile
export GAUSS_SRCDIR=$SLURM_SUBMIT_DIR

time g09 trial2.com




