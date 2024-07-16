#!/bin/bash
#SBATCH --job-name=s4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=s4.out

ml conda
conda activate evo
wandb agent tjbai/evolver/zada54jy
