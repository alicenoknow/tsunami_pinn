#!/bin/bash -l
#SBATCH -J an-pinn-0
#SBATCH --time=1:00:00 
#SBATCH -A plghailcanoon-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --output="slurm/output0.out"
#SBATCH --error="slurm/error0.err"
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu
#SBATCH  --mail-type=END,FAIL
module load cuda

cd $SLURM_SUBMIT_DIR

conda activate pinn_env
python run.py --config ./run_configs/mesh.json --run_num 0
