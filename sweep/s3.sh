#!/bin/bash
#SBATCH --job-name=s3
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=s3.out

ml conda
conda activate evo
<<<<<<< Updated upstream
=======
export REMOTE_PREFIX="/export/b12/tbai4"
>>>>>>> Stashed changes
wandb agent tjbai/evolver/zada54jy
