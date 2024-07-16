#!/bin/bash
#SBATCH --job-name=s3
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=s3.out

ml conda
conda activate evo
wandb agent zada54jy
