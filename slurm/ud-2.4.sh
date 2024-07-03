#!/bin/bash
#SBATCH --partition=a100
#SBATCH -A jeisner1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:0
#SBATCH --job-name=ud-2.4
#SBATCH --output=ud-2.4.out
#SBATCH --mem=80G

ml anaconda
conda activate evo
python3 train.py --config configs/ud-2.4.json --device cuda
