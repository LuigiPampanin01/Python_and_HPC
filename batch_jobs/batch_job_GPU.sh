#!/bin/sh
#BSUB -q c02613
#BSUB -J Saevar_final_12
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o GPU/Final_12_%J.out
#BSUB -e GPU/Final_12_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate_GPU.py 4751