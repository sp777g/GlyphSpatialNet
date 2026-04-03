#!/bin/bash
#SBATCH --job-name=FFG_eval_norm
#SBATCH --output=log_eval_norm.txt
#SBATCH --time=999:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=L40         # or 'gpujl'
#SBATCH --gres=gpu:1
CUDA_VISIBLE_DEVICES=0 conda run -n FFG --no-capture-output python evaluator_norm.py --dir results/test_UFSC >> res_UFSC_norm.txt 2>&1
CUDA_VISIBLE_DEVICES=0 conda run -n FFG --no-capture-output python evaluator_norm.py --dir results/test_UFUC >> res_UFUC_norm.txt 2>&1
echo "Job completed successfully."
