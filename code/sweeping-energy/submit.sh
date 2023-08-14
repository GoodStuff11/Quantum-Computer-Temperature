#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=gpt_test
#SBATCH --mem=8000M 
#SBATCH --gres=gpu:1 
#SBATCH --output=gpt_test-%J.out
#SBATCH --account=def-rgmelko
#SBATCH --cpus-per-task=1

module purge
# module load cuda cudnn  

# source /home/jkambulo/projects/def-rgmelko/jkambulo/py10/bin/activate
module load python/3.10

# export NCCL_BLOCKING_WAIT=1

python rydberg_rnn.py