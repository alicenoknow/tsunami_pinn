#!/bin/bash -l
#SBATCH -J an-pinn-test-relo
#SBATCH --time=05:00:00 
#SBATCH -A plghailcanoon-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --output="slurm/output4.out"
#SBATCH --error="slurm/error4.err"
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --gres=gpu
#SBATCH  --mail-type=END,FAIL
module load cuda

cd $SLURM_SUBMIT_DIR

nvidia-smi