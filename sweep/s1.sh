#!/bin/bash
#SBATCH --job-name=s1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=s1.out

ml conda
conda activate evo
wandb agent tjbai/evolver/zada54jy
