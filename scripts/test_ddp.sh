#!/bin/bash
#SBATCH -A m4789_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

ml conda
conda activate evo
torchrun test_ddp.py

~
~
~
~
~
~
~
~
~
~
