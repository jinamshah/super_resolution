#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
module load cuda/10.2.0
module load anaconda3

cd /jet/home/jshah2/super_resolution/vdsr
conda activate gpu

python validate.py vdsr_unpruned_2.pth.tar
python validate.py vdsr_unpruned_3.pth.tar
python validate.py vdsr_unpruned_4.pth.tar