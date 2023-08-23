#!/bin/bash
#SBATCH --mem=40G
#SBATCH --account=def-rgmelko
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=jkambulo@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --output=JSD_%J.out
#SBATCH --cpus-per-task=1

module purge
module load python/3.10

cd ~/projects/def-rgmelko/jkambulo/code/qc-temp
python ./compute_all_jsd.py