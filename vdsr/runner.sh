#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/10.2.0
module load anaconda3

cd /jet/home/jshah2/super_resolution/vdsr
conda activate gpu

# to generate dataset
cd scripts/ && python run.py
cd ../

# Data folders Urban100 and T91 are needed from here: https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD

python main.py

rm -rf data/Urban100/VDSR/