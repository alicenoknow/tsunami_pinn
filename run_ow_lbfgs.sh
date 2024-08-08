#!/bin/bash -l
#SBATCH -J an-pinn-underwater_lbfgs
#SBATCH --time=10:00:00 
#SBATCH -A plghailcanoon-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --output="slurm/output_underwater_lbfgs.out"
#SBATCH --error="slurm/error_underwater_lbfgs.err"
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu
#SBATCH  --mail-type=END,FAIL
module load cuda

cd $SLURM_SUBMIT_DIR

conda activate pinn_env
python over_lbfgs.py