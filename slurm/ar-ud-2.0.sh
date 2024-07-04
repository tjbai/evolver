#!/bin/bash
#SBATCH --partition=a100
#SBATCH -A jeisner1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:0
#SBATCH --job-name=ar-ud-2.0
#SBATCH --output=ar-ud-2.0.out
#SBATCH --mem=80G

ml anaconda
conda activate evo
python3 train.py --config configs/ar-ud-2.0.json --device cuda
