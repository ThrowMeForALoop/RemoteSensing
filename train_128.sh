#!/bin/sh
#SBATCH --time=47:00:00
#SBATCH -N 1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

. /etc/bashrc
. /etc/profile.d/modules.sh

module load cuda80/toolkit
module load cuDNN/cuda80/6.0.21

python train_128.py
