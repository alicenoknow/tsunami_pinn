#!/bin/bash -l
#SBATCH -J an-pinn-overwater_1
#SBATCH --time=10:00:00 
#SBATCH -A plghailcanoon-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --output="slurm/output_overwater_1.out"
#SBATCH --error="slurm/error_overwater_1.err"
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu
#SBATCH  --mail-type=END,FAIL
module load cuda

cd $SLURM_SUBMIT_DIR

conda activate pinn_env
python overwater.py