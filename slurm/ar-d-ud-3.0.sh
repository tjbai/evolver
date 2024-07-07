#!/bin/bash
#SBATCH --partition=a100
#SBATCH -A jeisner1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:0
#SBATCH --job-name=ar-d-ud-3.0
#SBATCH --output=ar-d-ud-3.0.out
#SBATCH --mem=80G

ml anaconda
conda activate evo
python3 train.py --config configs/ud-3/ar-d-ud-3.0.json --device cuda
