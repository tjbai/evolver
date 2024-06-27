#!/bin/bash

#SBATCH --partition=a100
#SBATCH -A jeisner1_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:0
#SBATCH --job-name=ar-d-ud-2.1.0
#SBATCH --output=ar-d-ud-2.1.0.out
#SBATCH --mem=80G

ml anaconda
conda activate evo

if [ ! -d "/scratch4/jeisner1/ar-d-ud-2.1.0" ]; then
  mkdir -p /scratch4/jeisner1/ar-d-ud-2.1.0
fi

python3 train.py ar_denoising \
    --train data/ud/ud_train_2.1.0.jsonl \
    --eval data/ud/ud_dev_2.1.0.jsonl \
    --config configs/ar-d-ud-2.1.0.json \
    --prefix ar-d-ud-2.1.0 \
    --device cuda
