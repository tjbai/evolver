#!/bin/bash

#SBATCH --partition=a100
#SBATCH -A jeisner1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:0
#SBATCH --job-name=ud-1.0.0
#SBATCH --output=ud-1.0.0.out
#SBATCH --mem=80G

ml anaconda
conda activate evo

python3 train.py \
    --train data/ud/ud.jsonl \
    --eval data/ud/en_ewt-ud-dev.conllu \
    --config configs/ud.json \
    --prefix ud-1.0.0 \
    --device cuda
