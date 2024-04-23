#!/bin/bash -l
#SBATCH -J an-pinn-test
#SBATCH --time=05:00:00 
#SBATCH -A plghailcanoon-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --output="slurm/output.out"
#SBATCH --error="slurm/error.err"
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu
#SBATCH  --mail-type=END,FAIL
module load cuda

cd $SLURM_SUBMIT_DIR

conda activate pinn_env
python run.py --config ./run_configs/mesh.json --run_num 0 --epochs 1000
