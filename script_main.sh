#!/bin/bash
#SBATCH --job-name=132_FULL
#SBATCH --output=log_main.txt
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=L40         # or 'gpujl'
#SBATCH --gres=gpu:1
conda run -n FFG --no-capture-output python main.py
echo "Job completed successfully."
