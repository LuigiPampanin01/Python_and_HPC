#!/bin/bash
#BSUB -J Parallell_timing
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -o Parallell_%J.out
#BSUB -e Parallell_%J.err
#BSUB -R "select[model == XeonGold6126]"

# Load environment and activate Conda
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run the script with the expected input file
python FloorPlanParStatic.py

