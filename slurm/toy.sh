#!/bin/bash

#SBATCH --partition=a100
#SBATCH -A jeisner1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:0
#SBATCH --job-name="toy"
#SBATCH --output=toy.out
#SBATCH --mem=32G

ml anaconda
conda activate ddm

python3 train.py \
    --train ../train/toy.jsonl \
    --eval ../train/toy_eval.txt \
    --config ./configs/toy.json \
    --prefix toy \
    --device cuda
