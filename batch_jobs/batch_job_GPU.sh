#!/bin/sh
#BSUB -q c02613
#BSUB -J Running_GPU_9
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o Running_GPU_9_%J.out
#BSUB -e Running_GPU_9_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python assignment9.py 20