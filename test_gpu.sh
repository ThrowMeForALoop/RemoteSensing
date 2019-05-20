#!/bin/sh

. /etc/bashrc
. /etc/profile.d/modules.sh


module load cuda80/toolkit
module load cuDNN/cuda80/6.0.21

python test_tensorflow.py
