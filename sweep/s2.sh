#!/bin/bash
#SBATCH --job-name=s2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=s2.out

ml conda
conda activate evo
wandb agent zada54jy
