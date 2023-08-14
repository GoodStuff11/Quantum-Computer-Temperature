#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=sweep_testing
#SBATCH --mem=20G
#SBATCH --gres=gpu:1 
#SBATCH --output=gpt_test-%J.out
#SBATCH --account=def-rgmelko
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4 
#SBATCH --nodes=1

module purge
module load cuda cudnn  

# source /home/jkambulo/projects/def-rgmelko/jkambulo/py10/bin/activate
module load python/3.10

# export NCCL_BLOCKING_WAIT=1

python main.py $1